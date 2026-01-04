### WMMA FlashAttention-2

A re-implementation of FlashAttention-2 forward pass using WMMA.

#### design

To resolve the issue of opaque wmma fragments, warp maximum is used instead of the original fa2's row maximum in controlling numerical stability of exponentials. Each block consists of a single warp and operates on 16 rows of $Q$ with $d = 64$.

#### demo

run in colab <br>

```bash
!pip install ninja
!git clone https://github.com/zack041/wmma-flashattention-v2
%cd wmma-flashattention-v2
!python benchmark/benchmark.py
```

#### benchmark
nvidia T4 using cuda.Event.record()

```bash
======================================================================
     N |          FA2 |         SDPA | Sanity Check
======================================================================
    64 |      0.046ms |      0.039ms | Truex
   128 |      0.089ms |      0.067ms | Truex
   256 |      0.231ms |      0.150ms | Truex
   512 |      0.846ms |      0.464ms | Truex
  1024 |      2.046ms |      1.107ms | Truex
  2048 |      4.512ms |      2.898ms | Truex
```
64% of SDPA!🎉
