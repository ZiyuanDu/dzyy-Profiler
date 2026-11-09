"""Profile TileLang GEMM and PyTorch with nsys or ncu."""

from pathlib import Path
from runpy import run_path

import torch
from torch.cuda import nvtx, profiler

# Compile the configuration to study.
M = N = K = 1024
gemm = run_path(str(Path(__file__).resolve().parents[1] / "00-basic-gemm" / "gemm.py"))["gemm"]
kernel = gemm.compile(
    M=M, N=N, K=K, BM=128, BN=64, BK=64,
    THREADS=256, STAGES=4,
)

# Prepare inputs and output buffers outside the capture.
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
a = torch.randn((M, K), device="cuda", dtype=torch.float16)
b = torch.randn((K, N), device="cuda", dtype=torch.float16)
c = torch.empty((M, N), device="cuda", dtype=torch.float16)
ref = torch.empty_like(c)
cache = torch.empty(256 * 1024 * 1024 // 4, device="cuda", dtype=torch.int32)
cache.zero_()
calls = (
    ("TileLang", lambda: kernel(a, b, c)),
    ("PyTorch", lambda: torch.mm(a, b, out=ref)),
)

# Warm up and validate before profiling.
for _, fn in calls:
    for _ in range(25):
        fn()
torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)
torch.cuda.synchronize()

# Capture GEMM calls; keep cache clearing outside the GEMM labels.
with profiler.profile():
    for _ in range(5):
        for name, fn in calls:
            cache.zero_()
            with nvtx.range("GEMM"), nvtx.range(name):
                fn()
    torch.cuda.synchronize()
