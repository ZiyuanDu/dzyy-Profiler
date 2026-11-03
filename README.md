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
- [Why Go Deeper into Profiling](#why-go-deeper-into-profiling)
- [TileLang, CUTLASS, CuTe, and Triton](#tilelang-cutlass-cute-and-triton)
- [Goals and Scope](#scope)
- [Curriculum](#curriculum)
- [GEMM Version Roadmap](#gemm-versions)

<a id="环境与运行"></a>

## Quick Start

Check the GPU, CUDA compiler, and host C++ compiler:

```bash
nvidia-smi
nvcc --version
g++ --version
```

On Ubuntu, install the host compiler if needed:

```bash
sudo apt update
sudo apt install -y build-essential
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

From the repository root:

```bash
uv sync --locked
```

`uv sync` creates a virtual environment in `.venv/` and installs the dependencies needed to run this project.

Use `uv run` for project commands:

```bash
uv run --locked 01-GEMM/00_basic_gemm.py
```

Look for **`correctness passed`**, followed by TileLang and PyTorch timings. The first run includes JIT compilation; later runs can reuse the cache.


## Goals and Scope

By the end of the series, you should be able to:

1. Translate an operator's equations, shapes, and layout into a working kernel.
2. Choose fusion boundaries and account for intermediate tensors and launches.
3. Analyze memory traffic, matrix instructions, synchronization, and resource use.
4. Build reliable benchmarks and test profiler-driven hypotheses.
5. Verify improvements across complete Attention, MoE, and hybrid execution paths.
