import argparse

import tilelang
import tilelang.language as T
import torch
from tilelang.profiler import do_bench


# Core: C = A @ B
@tilelang.jit
def gemm(A, B, C, BM=128, BN=128, BK=32, THREADS=128, STAGES=3):
    # Shape, tile, threads and pipeline stages are compile-time parameters.
    M, N, K = T.const("M, N, K")
    A: T.Tensor((M, K), T.float16)
    B: T.Tensor((K, N), T.float16)
    C: T.Tensor((M, N), T.float16)

    with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=THREADS) as (bx, by):
        # Reuse input tiles in shared memory; keep partial sums in registers.
        A_shared = T.alloc_shared((BM, BK), T.float16)
        B_shared = T.alloc_shared((BK, BN), T.float16)
        C_acc = T.alloc_fragment((BM, BN), T.float32)
        T.clear(C_acc)

        # Reduce over K in chunks of BK, overlapping tile loads and computation.
        for k in T.Pipelined(T.ceildiv(K, BK), num_stages=STAGES):
            # T.copy copies a whole tile, using the shared buffer's shape.
            T.copy(A[by * BM, k * BK], A_shared)
            T.copy(B[k * BK, bx * BN], B_shared)
            T.gemm(A_shared, B_shared, C_acc)

        # Store once after the full reduction
        T.copy(C_acc, C[by * BM, bx * BN])


# Runner: setup, validation, and timing. Skip this to focus on the kernel.
def main():
    parser = argparse.ArgumentParser(description="FP16 GEMM with FP32 accumulation.")
    parser.add_argument(
        "--shape",
        nargs=3,
        type=int,
        default=(1024, 1024, 1024),
        metavar=("M", "N", "K"),
        help="Matrix dimensions (default: 1024 1024 1024)",
    )
    parser.add_argument(
        "--source", action="store_true", help="Print generated CUDA source"
    )
    args = parser.parse_args()
    M, N, K = args.shape
    if min(M, N, K) <= 0:
        parser.error("M, N, and K must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU unavailable. Check NVIDIA drivers and GPU access.")

    torch.manual_seed(0)
    # Keep the PyTorch reference's reductions in FP32 for a comparable check.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)
    ref = torch.empty_like(c)

    # Compile and check correctness before timing; reuse both output buffers.
    kernel = gemm.compile(M=M, N=N, K=K)
    kernel(a, b, c)
    torch.mm(a, b, out=ref)
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)
    print(f"{torch.cuda.get_device_name()} | M={M}, N={N}, K={K} | correctness passed")

    # Compare median latency and throughput; p10/p90 show timing variation.
    for name, fn in (
        ("TileLang", lambda: kernel(a, b, c)),
        ("PyTorch", lambda: torch.mm(a, b, out=ref)),
    ):
        p10_ms, median_ms, p90_ms = do_bench(
            fn, backend="event", quantiles=[0.1, 0.5, 0.9]
        )
        # do_bench returns milliseconds; count each multiply-add as two FLOPs.
        tflops = 2 * M * N * K / median_ms / 1e9
        print(
            f"{name:8s}: {median_ms * 1e3:.2f} us | {tflops:.2f} TFLOP/s "
            f"| p10/p90={p10_ms * 1e3:.2f}/{p90_ms * 1e3:.2f} us"
        )
    if args.source:
        print(kernel.get_kernel_source())


if __name__ == "__main__":
    main()
