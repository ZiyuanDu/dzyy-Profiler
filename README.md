# Advanced GPU Kernels from Scratch

## A Quick Note

Hi everyone, long time no see!

I'm Du Ziyuan, a Bilibili creator and tech blogger. I share what I learn about AI infrastructure, GPU programming, and making large language models run faster.

I'm switching to English to reach more readers around the world and help you get more comfortable with technical English. Let's keep learning and sharing together!

Our earlier series covered the basics. This one moves into **kernel fusion, performance analysis, and modern LLM workloads**. Familiar operations such as RMSNorm, RoPE, Softmax, and SwiGLU become ingredients in larger optimization problems.

We'll use TileLang as the main implementation language, read other kernel ecosystems when they offer useful ideas, and follow one path throughout:

> **Math → Implementation → GPU execution → Profiling → Optimization**

Ready? Let's get started!

## Contents

- [Quick Start](#quick-start)
- [Environment Setup](00-Setup/README.md)
- [Why Go Deeper into Profiling](#why-go-deeper-into-profiling)
- [TileLang, CUTLASS, CuTe, and Triton](#tilelang-cutlass-cute-and-triton)
- [Goals and Scope](#scope)
- [Curriculum](#curriculum)
- [GEMM Version Roadmap](#gemm-versions)

<a id="环境与运行"></a>

## Quick Start

The introductory environment is managed by **uv**:

| Component | Version |
| --- | --- |
| Python | 3.12.13 |
| TileLang | 0.1.13 |
| PyTorch | 2.13.0 with CUDA 13.0 |

You'll need **Linux x86-64, an NVIDIA GPU, a CUDA 13-compatible driver, and an available CUDA Toolkit / `nvcc`**. TileLang and PyTorch are the direct dependencies; `uv.lock` pins the environment, and `.venv/` holds the virtual environment.

From the course repository root:

```bash
uv sync --locked
uv run 01-GEMM/00_basic_gemm.py
```

The first lesson starts with a [GEMM kernel of about 20 lines](01-GEMM/00_basic_gemm.py). It explains shapes, blocks, shared memory, fragments, the K loop, and the Python call. The example checks correctness against PyTorch and benchmarks both implementations with the same timing method. Follow the [step-by-step walkthrough](01-GEMM/00-basic-gemm.md) alongside the code.

For installation steps, environment checks, and common fixes, follow [Environment Setup](00-Setup/README.md). Later chapters using newer APIs or architecture-specific instructions will state their own tested requirements.

## Why Go Deeper into Profiling

A correct kernel answers whether we computed the right result. A timer tells us how long it took. **Profiling helps explain what to change next.**

### Find the Limiting Resource

The same operation can behave very differently during prefill and decode. A large GEMM may have enough work to keep Tensor Cores busy; a small workload may spend proportionally more time moving data, launching work, or waiting on dependencies.

We need to connect the workload to evidence:

- **Memory traffic:** Are intermediate tensors written to device memory and immediately read back? Can we reuse data in registers or shared memory?
- **Compute and scheduling:** Are matrix instructions active? Is there enough parallel work? Do layout conversions or synchronization leave gaps?
- **Resource pressure:** Does a larger tile or deeper pipeline increase register use, spills, or shared memory enough to reduce useful concurrency?
- **Execution overhead:** Are many short kernels separated by CPU or launch gaps? Does the bottleneck sit outside the kernel we are tuning?

[Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) helps locate important regions on the CPU/GPU timeline. [Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) provides kernel metrics, resource analysis, and roofline views. We will use each tool to answer a specific question, then verify the conclusion against generated code and independent timing.

### Understand When Fusion Helps

Consider a projection followed by bias and an activation. Separate kernels materialize the projection output and read it again. An epilogue can apply the extra work before the result is stored.

For one eliminated intermediate tensor containing `M * N` elements, a simple traffic estimate is **one write plus one read**, or `2 * M * N * sizeof(dtype)` bytes. That is an accounting estimate, not a predicted speedup: caches, remaining accesses, and execution overhead still matter.

Fusion also changes the schedule. It can lengthen register lifetimes, introduce reductions, or prevent an otherwise efficient GEMM configuration. We will compare **unfused, partially fused, and fused paths**, including cases where fusion loses.

> The useful question is how much time the complete operation saves, and what explains that result.

### Connect Kernel Gains to the Whole Block

Suppose a kernel accounts for 30% of block latency. Making it twice as fast gives an idealized block speedup of `1 / (0.7 + 0.3 / 2) ≈ 1.18×`, assuming the remaining work is unchanged. The next optimization may belong in dispatch, cache access, or another stage.

An official [PyTorch benchmark report](https://pytorch.org/blog/gemms-torchinductor-cutedsl-backend/) illustrates the difference:

- **Kernel result:** up to **1.73×** improvement on selected BF16 decode GEMMs.
- **Application result:** a maximum **6.5%** BF16 inference latency reduction, on Llama 3.3 70B at batch size 16.

These results describe the report's workloads; they are not expected gains for this course.

<details class="reading-note">
<summary>Benchmark conditions</summary>

The April 2026 report used one B200 at 850 W with dynamic clocks, PyTorch nightly, and CUDA 13.1. Inference used vLLM V1 with 32 input tokens and 128 generated tokens.

</details>

### Make Each Experiment Explainable

Every optimization experiment will record the GPU and software versions, shapes and layouts, precision, timing scope, warmup, repeated measurements, and numerical error. Compilation, steady-state execution, and profiling runs will be reported separately.

Our loop is **baseline → evidence → hypothesis → one controlled change → measurement → explanation**. A lower runtime is a result; understanding when and why it occurs is the skill we are developing.

## TileLang, CUTLASS, CuTe, and Triton

These tools occupy different levels of the kernel stack. **CuTe is part of the CUTLASS ecosystem**: CuTe C++ provides layout and tensor abstractions, while CuTe DSL exposes related concepts through Python. Treating all four names as independent competing products hides that relationship. [NVIDIA CuTe guide](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/00_quickstart.html), [CuTe DSL introduction](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html)

### What Each Tool Gives Us

| Tool | Programming focus | Role in this series |
| --- | --- | --- |
| TileLang | Python-style tiles, explicit storage scopes, data movement, matrix operations, and pipelines | Main implementation language for controlled optimization experiments |
| CUTLASS | NVIDIA's C++ templates and kernel components for GEMM and related operations, including fused epilogues | Reference designs for Tensor Core kernels, scheduling, and epilogue fusion |
| CuTe C++ / CuTe DSL | Layout algebra, tensor partitioning, copy/MMA primitives, and explicit hardware organization | Deeper study of thread/data mapping and architecture-specific implementations |
| Triton | Tensor-block programming, reductions, fusion, and compiler-managed mapping; Gluon exposes more explicit control | Comparative implementations and a connection to compiler-generated kernels |

The descriptions follow the [TileLang overview](https://tilelang.com/get_started/overview.html), [CUTLASS documentation](https://docs.nvidia.com/cutlass/latest/overview.html), [CuTe documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/00_quickstart.html), and [Triton/Gluon documentation](https://triton-lang.org/main/gluon/index.html). Performance depends on the workload and implementation, so comparisons will use matching semantics, shapes, precision, and hardware.

### Public Usage and Community Signals

The following snapshot was checked on **November 3, 2026**. Counts are rounded as displayed by GitHub. Stars and forks measure repository interest, not production deployments or market share.

| Official repository | Stars | Forks |
| --- | --- | --- |
| [TileLang](https://github.com/tile-ai/tilelang) | ~8.9k | 951 |
| [CUTLASS, including CuTe](https://github.com/NVIDIA/cutlass) | ~10.7k | ~2.2k |
| [Triton](https://github.com/triton-lang/triton) | ~20.5k | ~3.4k |

CuTe shares the CUTLASS repository, so it has no separate count here. More useful evidence comes from concrete projects:

- **TileLang:** DeepSeek's TileKernels authors report internal training and inference use. Its public kernels include MoE routing, quantization, fused SwiGLU/quantization, Engram gating, and mHC. [DeepSeek TileKernels](https://github.com/deepseek-ai/TileKernels)
- **CUTLASS and CuTe DSL:** PyTorch documents CUTLASS C++ as a GEMM autotuning backend and the addition of a CuTe DSL backend. The April report treats further integration work, including its CuTe DSL epilogue-fusion path, as ongoing. [PyTorch integration report](https://pytorch.org/blog/gemms-torchinductor-cutedsl-backend/)
- **Triton:** The same PyTorch report describes its role in GEMM autotuning and pointwise/reduction workloads. Flash Linear Attention provides public implementations for sequence models such as Gated DeltaNet. [PyTorch backend discussion](https://pytorch.org/blog/gemms-torchinductor-cutedsl-backend/), [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)

These examples establish real development and integration paths. They do not establish a universal winner or an industry adoption percentage.

### Why TileLang Is Our Starting Point

TileLang makes the data path visible without requiring us to write every thread-coordination detail. In a compact GEMM:

- **Allocate:** `T.alloc_shared` and `T.alloc_fragment` declare storage.
- **Move:** `T.copy` transfers tiles.
- **Compute:** `T.gemm` expresses tile matrix multiplication.
- **Overlap:** `T.Pipelined` organizes a software pipeline.

That structure follows the questions we want to investigate: where data lives, how it is reused, and whether movement overlaps with computation. Layout, tile size, thread count, and pipeline depth become variables we can change and measure. See the [official GEMM example](https://github.com/tile-ai/tilelang/blob/main/examples/gemm/example_gemm.py).

We will read CUTLASS/CuTe and Triton implementations where they clarify an optimization. Learning the GPU execution model makes that knowledge transferable between languages.

<a id="scope"></a>

## Goals and Scope

By the end of the series, you should be able to:

1. Translate an operator's equations, shapes, and layout into a working kernel.
2. Choose fusion boundaries and account for intermediate tensors and launches.
3. Analyze memory traffic, matrix instructions, synchronization, and resource use.
4. Build reliable benchmarks and test profiler-driven hypotheses.
5. Verify improvements across complete Attention, MoE, and hybrid execution paths.

The main track covers **forward computation, prefill, and decode**, starting on a single GPU. We assume Python, PyTorch, basic GPU programming, and familiarity with common operators from the earlier series. Backpropagation and distributed communication are later extensions.

Recent case studies include DeepSeek-style sparse Attention, mHC, and Engram, alongside Gated DeltaNet. The [DeepSeek sparse-attention release](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp#open-source-kernels), [TileKernels](https://github.com/deepseek-ai/TileKernels), and [FLA implementations](https://github.com/fla-org/flash-linear-attention) provide public reference points. Each lesson will select a concrete operator and define its semantics before optimization.

## Curriculum

The main track contains **20 chapters in five parts**. A typical chapter compares 2–3 implementations across representative shapes, with 1–2 cases for deeper profiling. Basic operations are reused inside fused paths; the emphasis is optimization and its effect on complete workloads.

<a id="part-gemm"></a>

### Part 1 GEMM as an Optimization Laboratory

| Chapter | Topics | Experiment and profiling focus |
| --- | --- | --- |
| 01 A GEMM We Can Measure | JIT, correctness, matched timing, generated code | Separate compilation, launch overhead, and steady-state execution |
| 02 Tiles, Layouts, and Tensor Cores | Shared-memory reuse, fragments, matrix instructions, shape effects | Relate tile choices to data movement and resource use |
| 03 Pipelines and Scheduling | Load/compute overlap, stages, warp specialization where supported | Check whether overlap improves without excessive resource pressure |
| 04 GEMM Epilogue Fusion | Bias, activation, scaling, residual updates | Compare unfused, partially fused, and fused complete operations |
| 05 Profiling-Guided Tuning | Tile/thread/stage search, layout experiments, shape dispatch | Validate bottleneck hypotheses and track tuning cost and regressions |

<a id="part-dense"></a>

### Part 2 Fusion Across LLM Execution Paths

| Chapter | Topics | Experiment and profiling focus |
| --- | --- | --- |
| 06 Residual and Normalization Fusion | Reuse existing add/RMSNorm implementations; investigate adjacent layout work | Count eliminated intermediates and test launch-limited versus bandwidth-limited shapes |
| 07 FFN Fusion Boundaries | Gate/up projection, SwiGLU epilogues, output projection | Find profitable fusion boundaries and measure the full FFN path |
| 08 Quantization and Matmul Fusion | Scaling, casting, dequantization, FP8; selected FP4 extensions | Include metadata traffic and conversion cost; validate accuracy and hardware requirements |

<a id="part-attention"></a>

### Part 3 Attention and the KV Path

| Chapter | Topics | Experiment and profiling focus |
| --- | --- | --- |
| 09 Fused Attention Prefill | Online softmax, tiled QK/PV, masks, GQA | Measure avoided intermediates and the impact of sequence/head shapes |
| 10 Decode and Paged KV | RoPE/cache-write fusion, split-KV, paged reads, result merging | Compare contiguous and paged layouts, context lengths, and batch sizes |
| 11 MLA Decode | Latent cache, absorbed projections, fused attention computation | Relate cache savings to compute, register pressure, and total decode latency |
| 12 Sparse Attention and Indexing | DeepSeek-style index scores, Top-K selection, gather, sparse attention | Include selection and data-access costs in the complete sparse path |

<a id="part-moe"></a>

### Part 4 MoE from Routing to Combination

| Chapter | Topics | Experiment and profiling focus |
| --- | --- | --- |
| 13 Fused Router and Top-K | Scores, expert selection, model-specific normalization | Reduce temporary buffers while preserving routing semantics |
| 14 Dispatch and Expert Layouts | Counts, offsets, permutation, inverse mapping | Measure reordering with empty experts and uneven token assignments |
| 15 Grouped and Quantized GEMM | Variable expert sizes, shared scheduling, quantized paths | Compare per-expert launches, padding, grouped execution, and conversion overhead |
| 16 MoE Fusion and Combination | Activation/quantization boundaries, weighted merge, shared experts | Check whether expert-kernel gains survive dispatch and combination costs |

<a id="part-hybrid"></a>

### Part 5 Emerging Operators and Complete Blocks

| Chapter | Topics | Experiment and profiling focus |
| --- | --- | --- |
| 17 Gated DeltaNet | Recurrent updates, fused gates, state layout; chunked extensions | Analyze state traffic, parallelism, and long-sequence numerical error |
| 18 mHC | Small mixing operations, normalization, residual-stream application | Investigate fusion and synchronization in small, frequent operations |
| 19 Engram Gating | Lookup-fed features, normalization, gating, fusion boundaries | Account for irregular data access and distinguish gating from the full memory path |
| 20 Complete Block Validation | Reuse optimized Attention/MoE/hybrid components in 2–4 small layers | Check numerical agreement and profile the complete execution path |

The final assembly exercise uses random weights and small configurations to validate integration and latency composition. Model quality evaluation requires a separate experiment.

<a id="gemm-versions"></a>

## GEMM Version Roadmap

The first optimization sequence keeps the original version names. Each version isolates one main change; fusion extends the operation and uses a reference with matching semantics.

| Version | Main change | Question to answer |
| --- | --- | --- |
| `v00_simt` | Small direct-computation reference | Where do redundant accesses occur? |
| `v01_shared` | Tile reuse in shared memory | Does reuse justify movement and synchronization? |
| `v02_tensorcore` | Use `T.gemm` | Which matrix instructions execute? |
| `v03_pipeline` | Overlap loading and computation | Is overlap useful at the resulting resource cost? |
| `v04_tuned_tiles` | Tune tiles, threads, and stages | Which configurations suit each workload? |
| `v05_layout` | Adjust layouts or block scheduling | Does the generated code change in the intended way? |
| `v06_epilogue` | Fuse postprocessing | Does the complete operation become faster? |
| `v07_dispatch` | Select by shape | Do improvements hold across representative workloads? |

The SIMT reference is a small teaching exercise. Later comparisons use a shared workload set and equivalent full operations. Manual settings may produce identical code or lower performance; those results belong in the record too. The [software pipeline guide](https://tilelang.com/programming_guides/software_pipeline.html) provides additional implementation context.
