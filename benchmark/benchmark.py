import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

wmma_fa2 = load(
    name='wmma_fa2',
    sources=['src/binding.cpp', 'kernels/wmma_fa2.cu'],
    verbose=True
)

def fa2_attention(Q,K,V):
    return wmma_fa2.forward(Q,K,V)

def sdpa_attention(Q,K,V):
    return torch.nn.functional.scaled_dot_product_attention(
        Q, K, V,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False
    )

def benchmark(f, Q, K, V):
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(10):
        f(Q,K,V) #warmup
        
    torch.cuda.synchronize()
    start.record()
    for _ in range(42):
        res = f(Q, K, V)
    end.record()
    torch.cuda.synchronize()
    
    return res, start.elapsed_time(end)/42

print("=" * 70)
print(f"{'N':>6} | {'FA2':>12} | {'SDPA':>12} | {'Sanity Check':>12}")
print("=" * 70)

Q = torch.randn(2, 4, 2048, 64, device='cuda')
K = torch.randn(2, 4, 2048, 64, device='cuda')
V = torch.randn(2, 4, 2048, 64, device='cuda')
fa2_attention(Q,K,V)
sdpa_attention(Q,K,V)

for N in [64, 128, 256, 512, 1024, 2048]:
    Q = torch.randn(2, 4, N, 64, device='cuda')
    K = torch.randn(2, 4, N, 64, device='cuda')
    V = torch.randn(2, 4, N, 64, device='cuda')
    fa2val, fa2time = benchmark(fa2_attention, Q, K, V)
    sdpaval, sdpatime = benchmark(sdpa_attention, Q, K, V)
    print(f"{N:>6} | {fa2time:>10.3f}ms | {sdpatime:>10.3f}ms | {torch.allclose(fa2val, sdpaval, atol=1e-3)}x")
