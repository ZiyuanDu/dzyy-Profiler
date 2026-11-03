"""FP16 GEMM with FP32 accumulation; run from the project root with uv."""
import argparse

import tilelang
import tilelang.language as T
import torch
from tilelang.profiler import do_bench


@tilelang.jit
def gemm(A, B, C, BM=128, BN=128, BK=32):
    M, N, K = T.const("M, N, K")
    A: T.Tensor((M, K), T.float16)
    B: T.Tensor((K, N), T.float16)
    C: T.Tensor((M, N), T.float16)

    with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
        A_shared = T.alloc_shared((BM, BK), T.float16)
        B_shared = T.alloc_shared((BK, BN), T.float16)
        C_acc = T.alloc_fragment((BM, BN), T.float32)
        T.clear(C_acc)
        for k in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
            T.copy(A[by * BM, k * BK], A_shared)
            T.copy(B[k * BK, bx * BN], B_shared)
            T.gemm(A_shared, B_shared, C_acc)
        T.copy(C_acc, C[by * BM, bx * BN])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        nargs=3,
        type=int,
        default=(1024, 1024, 1024),
        metavar=("M", "N", "K"),
    )
    parser.add_argument(
        "--source", action="store_true", help="Print generated CUDA source"
    )
    args = parser.parse_args()
    M, N, K = args.shape
    if min(M, N, K) <= 0:
        parser.error("M, N and K must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("需要可访问的 NVIDIA GPU；请检查驱动及当前进程的 GPU 权限。")

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)
    ref = torch.empty_like(c)
    kernel = gemm.compile(M=M, N=N, K=K)
    kernel(a, b, c)
    torch.mm(a, b, out=ref)
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)
    print(f"{torch.cuda.get_device_name()} | M={M}, N={N}, K={K} | correctness passed")

    for name, fn in (
        ("TileLang", lambda: kernel(a, b, c)),
        ("PyTorch", lambda: torch.mm(a, b, out=ref)),
    ):
        p10, median, p90 = do_bench(fn, backend="event", quantiles=[0.1, 0.5, 0.9])
        print(
            f"{name}: {median * 1e3:.2f} us (p10={p10 * 1e3:.2f}, p90={p90 * 1e3:.2f}), {2 * M * N * K / median / 1e9:.2f} TFLOP/s"
        )
    if args.source:
        print(kernel.get_kernel_source())


if __name__ == "__main__":
    main()
