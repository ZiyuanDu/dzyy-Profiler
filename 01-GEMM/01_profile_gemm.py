import argparse
from importlib import import_module

import torch
from torch.cuda import nvtx, profiler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        nargs=3,
        type=int,
        default=(1024, 1024, 1024),
        metavar=("M", "N", "K"),
        help="Matrix dimensions (default: 1024 1024 1024)",
    )
    args = parser.parse_args()
    M, N, K = args.shape
    if min(M, N, K) <= 0:
        parser.error("M, N, and K must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU unavailable. Check NVIDIA drivers and GPU access.")

    # Reuse the first lesson's kernel; keep compilation outside the capture.
    gemm = import_module("00_basic_gemm").gemm
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)
    ref = torch.empty_like(c)
    kernel = gemm.compile(M=M, N=N, K=K)
    calls = (
        ("TileLang", lambda: kernel(a, b, c)),
        ("PyTorch", lambda: torch.mm(a, b, out=ref)),
    )

    kernel(a, b, c)
    torch.mm(a, b, out=ref)
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)

    # Match do_bench's default 256 MiB int32 cache-flush buffer.
    cache = torch.empty(256 * 1024 * 1024 // 4, device="cuda", dtype=torch.int32)
    cache.zero_()
    # Initialize NVTX as well as the kernels before the capture starts.
    with nvtx.range("warmup"):
        for _, fn in calls:
            for _ in range(25):
                fn()
    torch.cuda.synchronize()
    print(
        f"{torch.cuda.get_device_name()} | M={M}, N={N}, K={K} | correctness passed\n"
        "Capture: 20 calls per implementation | flush before each call"
    )

    # --capture-range=cudaProfilerApi records only this warmed region.
    with profiler.profile():
        for name, fn in calls:
            for _ in range(20):
                cache.zero_()  # Keep benchmark preparation outside the GEMM label.
                with nvtx.range(name):
                    fn()
            torch.cuda.synchronize()

    print("Capture complete. Inspect the Nsight Systems report.")


if __name__ == "__main__":
    main()
