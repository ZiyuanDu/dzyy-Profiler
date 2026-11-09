# GEMM

[Back to the project](../README.md)

| Lesson | Topics | Code |
| --- | --- | --- |
| [00 · Basic GEMM](00-basic-gemm/README.md) | Tiling, threads, shared memory, and pipelining | [gemm.py](00-basic-gemm/gemm.py) |
| [01 · Profiling and Optimization](01-profiling/README.md) | Nsight Systems, Nsight Compute, and measured optimization | [profile_gemm.py](01-profiling/profile_gemm.py) |

Each lesson keeps its documentation, Python example, and figures together. Run the commands in both lessons from the project root.
# GEMM: From a First Kernel to LLM Workloads

We start with a working TileLang GEMM and build toward the kernels behind LLM linear layers and MoE experts. The goal is to understand the execution, explain the measurements, and choose an effective implementation for each GPU.

Modules **01–02** are available. Modules **03–17** form the planned learning path.

## 1. Build and Measure

| Module | What we will do |
| --- | --- |
| **01. [GEMM Basics](00-basic-gemm.md)** | Implement FP16 GEMM with FP32 accumulation in TileLang, understand its tiles and buffers, check correctness, and benchmark against PyTorch. |
| **02. [Profiling Basics](01-nsight-systems.md)** | Use nsys to find GPU work and ncu to inspect a kernel. Establish comparable measurements and test tile, thread, and pipeline settings. |
| **03. Roofline and the Memory Hierarchy** | Find the appropriate Tensor Core roofline in ncu, understand arithmetic intensity and compute ceilings, and relate measured memory traffic to attainable performance. |

## 2. Understand the Kernel

| Module | What we will do |
| --- | --- |
| **04. Memory Access and Layouts** | Study global memory coalescing, vectorized copies, shared memory layouts, bank conflicts, alignment, and partial tiles. Use memory metrics to explain the effects. |
| **05. Tensor Cores and Generated Code** | Follow `T.gemm` into generated CUDA, PTX, and SASS. Understand warp tiles, data fragments, accumulator registers, and the matrix instructions available on the target GPU. |
| **06. Asynchronous Loads and Pipelines** | Go beyond changing `STAGES`: examine asynchronous copies, barriers, buffer reuse, and producer–consumer warps. Measure how much loading overlaps computation. |
| **07. Resources and Ready Warps** | Connect registers, shared memory, spills, and occupancy with scheduler activity. Use eligible warps and stall evidence to test what limits instruction issue. |

## 3. Schedule, Fuse, and Specialize

| Module | What we will do |
| --- | --- |
| **08. Persistent Scheduling and Cache Reuse** | Build a persistent tile scheduler and compare tile traversal orders. Examine L2 reuse, work distribution, and the final wave of blocks. |
| **09. Split-K and Stream-K** | Distribute reduction work across blocks when output tiles provide too little parallelism. Measure the complete computation, including reductions, workspace traffic, and numerical effects. |
| **10. Fused Epilogues** | Fuse bias, residuals, activations, and scaling where appropriate. Compare the complete operation, including intermediate memory traffic and kernel launches. |
| **11. Autotuning and Dispatch** | Search valid configurations and build dispatch rules for shape, layout, dtype, and GPU architecture. Account for compilation and tuning costs separately from execution. |

## 4. Work with LLM GEMMs

| Module | What we will do |
| --- | --- |
| **12. Linear Layers in Real Models** | Extract projection and MLP GEMMs from model code such as [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3/tree/main/inference). Preserve layouts and dtypes, and compare realistic prefill, decode, and training workloads. |
| **13. FP8 and Block Scaling** | Implement scaling, quantization, and mixed-precision accumulation. Study scale layouts and numerical error, and measure both GEMM time and the cost of preparing its inputs. |
| **14. Quantized and Low-Bit GEMM** | Compare weight-only INT8/INT4 approaches and hardware-supported FP4 paths. Examine packing, dequantization, bandwidth savings, and complete linear-layer latency. |
| **15. Batched and Grouped GEMM for MoE** | Extend one GEMM to batches and groups, then handle uneven expert workloads. Include scheduling, token rearrangement, and metadata costs in the measurements. |

## 5. Study Production Kernels and Port the Method

| Module | What we will do |
| --- | --- |
| **16. DeepGEMM Case Studies** | Read a pinned [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) revision and reproduce selected dense and grouped GEMM designs in TileLang. Explain their layouts, pipelines, scaling, and scheduling with profiler evidence. |
| **17. GEMM Across GPU Generations** | Repeat selected case studies on available RTX and datacenter GPUs. Check instruction support, retune resource use, and compare against suitable libraries on each GPU with matching accuracy requirements. |

## How the Chapters Connect

Each optimization chapter extends a working baseline and follows the same process:

> Predict a hardware effect → change one design choice → check correctness and timing → inspect relevant profiler evidence → explain the result.

Use shapes from real workloads as test cases throughout the series. Record GPU architecture, software versions, clock and cache settings, and numerical tolerances so that results can be reproduced. For multi-kernel or quantized designs, report the cost of the complete operation as well as the GEMM kernel.

Roofline provides a performance model; the [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline) explains its charts and metrics. Use occupancy, scheduler, memory, and instruction evidence to investigate the gap between a measured point and its roofline.

## Working Across GPUs

The profiling labs use NVIDIA GPUs. Keep a common TileLang baseline, then select architecture-specific instructions and schedules where supported. The analysis process carries across GPUs; instruction paths and tuning choices must be checked for each device.

[DeepGEMM's current requirements](https://github.com/deepseek-ai/DeepGEMM#requirements) list SM90 and SM100 GPUs. [CUTLASS documents a separate SM120 GEMM path](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html#blackwell-sm120-gemms) for RTX 50-series GPUs. Pin the source revision and supported hardware for each case study, and use an appropriate reference implementation on each target.
