#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h> 

using namespace nvcuda;

/*
Br, Bc = 32, d = 64
*/

__inline__ __device__ float pairReduceSum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

__inline__ __device__ float warpReduceMax(float val) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 4));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 8));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 16));
    return val;
}
__global__ void wmma_fa2(float* Q, float* K, float* V, float* O, int N, int d, int Bc, int Br, int Tc, int Tr, float softmax_scale){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int head = blockIdx.y;
    int batch = blockIdx.z;
    int lanex = tx;

    int q_offset = batch*gridDim.y*N*d + head*N*d + 16*bx*d;
    int kv_offset = batch*gridDim.y*N*d + head*N*d;

    extern __shared__ char sram[];
    __half* Qi = (__half*)sram;
    __half* Kj = (__half*)&sram[2048];
    __half* Vj = (__half*)&sram[6144];
    float* S = (float*)&sram[10240];
    __half* Shalf = (__half*)&sram[12288];

    // fetching Qi
    #pragma unroll
    for(int i=0;i<32;i++){
        if (16 * bx + i/2 < N){
            Qi[lanex+32*i] = __float2half(Q[q_offset+lanex+32*i]);
        }
        else{
            Qi[lanex+32*i] = __float2half(0.0f);
        }
    }

    // initializing Oi, l, m
    float l = 0.0f;
    float m = -INFINITY;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> c_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c1_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c2_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> o1_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> o2_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> o3_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> o4_frag;
    wmma::fill_fragment(o1_frag, 0.0f);
    wmma::fill_fragment(o2_frag, 0.0f);
    wmma::fill_fragment(o3_frag, 0.0f);
    wmma::fill_fragment(o4_frag, 0.0f);

    for(int j=0; j<Tc; j++){
        __syncthreads();
        #pragma unroll
        for(int i=0;i<64;i++){
            if(j*Bc+i/2<N){
                Kj[i*32+lanex] = __float2half(K[kv_offset + j*Bc*64 + i*32 + lanex]);
                Vj[i*32+lanex] = __float2half(V[kv_offset + j*Bc*64 + i*32 + lanex]);
            }
            else{
                Kj[i*32+lanex] = __float2half(0.0f);
                Vj[i*32+lanex] = __float2half(0.0f);
            }
        }
        __syncthreads();

        //compute S
        wmma::fill_fragment(c1_frag, 0.0f);
        wmma::fill_fragment(c2_frag, 0.0f);

        __syncthreads();
        #pragma unroll
        for(int i=0;i<4;i++){
            wmma::load_matrix_sync(a_frag, &Qi[16*i], 64);
            wmma::load_matrix_sync(b_frag, &Kj[16*i], 64);
            wmma::mma_sync(c1_frag, a_frag, b_frag, c1_frag);
            wmma::load_matrix_sync(b_frag, &Kj[16*64+16*i], 64);
            wmma::mma_sync(c2_frag, a_frag, b_frag, c2_frag);
        }

        wmma::store_matrix_sync(&S[0], c1_frag, 32, wmma::mem_row_major);
        wmma::store_matrix_sync(&S[16], c2_frag, 32, wmma::mem_row_major);
        __syncthreads();
        
        float lanemax = -INFINITY;
        for(int i=0;i<16;i++){
            lanemax = max(lanemax,S[lanex*16+i]*softmax_scale);
        }
        float tilemax = warpReduceMax(lanemax);
        float mprev = m;
        m = fmaxf(m,tilemax);
        float update = __expf(mprev-m);
        l = update*l;
        float lanep = 0.0f;
        for(int i=0;i<16;i++){
            S[lanex*16+i] = __expf((S[lanex*16+i])*softmax_scale-m);
            lanep += S[lanex*16+i];
            Shalf[lanex*16+i] = __float2half(S[lanex*16+i]);
        }
        l += pairReduceSum(lanep);
        
        for(int i = 0; i < o1_frag.num_elements; i++){
            o1_frag.x[i] *= update;
            o2_frag.x[i] *= update;
            o3_frag.x[i] *= update;
            o4_frag.x[i] *= update;
        }

        __syncthreads();

        wmma::load_matrix_sync(a_frag, &Shalf[0], 32);
        wmma::load_matrix_sync(c_frag, &Vj[0], 64);
        wmma::mma_sync(o1_frag, a_frag, c_frag, o1_frag);
        wmma::load_matrix_sync(a_frag, &Shalf[16], 32);
        wmma::load_matrix_sync(c_frag, &Vj[16*64], 64);
        wmma::mma_sync(o1_frag, a_frag, c_frag, o1_frag);

        wmma::load_matrix_sync(a_frag, &Shalf[0], 32);
        wmma::load_matrix_sync(c_frag, &Vj[16], 64);
        wmma::mma_sync(o2_frag, a_frag, c_frag, o2_frag);
        wmma::load_matrix_sync(a_frag, &Shalf[16], 32);
        wmma::load_matrix_sync(c_frag, &Vj[16+16*64], 64);
        wmma::mma_sync(o2_frag, a_frag, c_frag, o2_frag);

        wmma::load_matrix_sync(a_frag, &Shalf[0], 32);
        wmma::load_matrix_sync(c_frag, &Vj[32], 64);
        wmma::mma_sync(o3_frag, a_frag, c_frag, o3_frag);
        wmma::load_matrix_sync(a_frag, &Shalf[16], 32);
        wmma::load_matrix_sync(c_frag, &Vj[32+16*64], 64);
        wmma::mma_sync(o3_frag, a_frag, c_frag, o3_frag);

        wmma::load_matrix_sync(a_frag, &Shalf[0], 32);
        wmma::load_matrix_sync(c_frag, &Vj[16*3], 64);
        wmma::mma_sync(o4_frag, a_frag, c_frag, o4_frag);
        wmma::load_matrix_sync(a_frag, &Shalf[16], 32);
        wmma::load_matrix_sync(c_frag, &Vj[16*3+16*64], 64);
        wmma::mma_sync(o4_frag, a_frag, c_frag, o4_frag);

        __syncthreads();
    }

    wmma::store_matrix_sync(&O[q_offset], o1_frag, 64, wmma::mem_row_major);
    wmma::store_matrix_sync(&O[q_offset+16], o2_frag, 64, wmma::mem_row_major);
    wmma::store_matrix_sync(&O[q_offset+32], o3_frag, 64, wmma::mem_row_major);
    wmma::store_matrix_sync(&O[q_offset+48], o4_frag, 64, wmma::mem_row_major);

    __shared__ float row_l[16];
    if (lanex % 2 == 0) {
        row_l[lanex / 2] = l;
    }
    __syncthreads();

    for(int i = 0; i < 32; i++) {
        O[q_offset + lanex+32*i] /= row_l[(lanex+32*i)/64];
    }
}

torch::Tensor fa2forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    int B = Q.size(0);
    int h = Q.size(1);
    int N = Q.size(2);
    int d = Q.size(3);
    
    int Bc = 32;
    int Br = 32;

    int Tc = (N+Bc-1)/Bc;
    int Tr = (N+Br-1)/Br;

    float softmax_scale = 1.0f/sqrtf((float)d);

    dim3 grid_dim(N/16,h,B);
    dim3 block_dim(Br);
    int sram_size = (16*64*2 + 2*32*64*2 + 16*32*4 + 16*32*2); //Q,O,K,V,S,Shalf

    torch::Tensor O = torch::empty({B, h, N, d}, Q.options());

    wmma_fa2<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), 
        O.data_ptr<float>(), N, d, Bc, Br, Tc, Tr, softmax_scale);

    return O;
}
