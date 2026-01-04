import torch
from torch.utils.cpp_extension import load

wmma_fa2 = load(
    name='wmma_fa2',
    sources=[
        'src/binding.cpp',
        'kernels/wmma_fa2.cu',
    ],
    verbose=True
)

def attention(Q, K, V):
    """
    Q, K, V: [B, H, N, d]
    Return O: [B, H, N, d]
    """
    return wmma_fa2.forward(Q, K, V)


Q = torch.randn(2, 4, 128, 64, device='cuda')
K = torch.randn(2, 4, 128, 64, device='cuda')
V = torch.randn(2, 4, 128, 64, device='cuda')

O = attention(Q, K, V)
print(O)
