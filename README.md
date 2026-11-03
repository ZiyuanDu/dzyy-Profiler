<p align="center">
  <img src="assets/dzyy-profiler-icon.png" alt="dzyy-Profiler logo" width="128" />
</p>

<h1 align="center">dzyy-Profiler</h1>

<p align="center">GPU kernel profiling, benchmarking, and optimization.</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&amp;logoColor=white" alt="Python 3.12" /></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.13.0-EE4C2C?logo=pytorch&amp;logoColor=white" alt="PyTorch 2.13.0" /></a>
  <a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-13.0-76B900?logo=nvidia&amp;logoColor=white" alt="CUDA 13.0" /></a>
  <a href="https://github.com/tile-ai/tilelang"><img src="https://img.shields.io/badge/TileLang-0.1.13-0F766E" alt="TileLang 0.1.13" /></a>
  <a href="https://github.com/ZiyuanDu/dzyy-Profiler/stargazers"><img src="https://img.shields.io/github/stars/ZiyuanDu/dzyy-Profiler?style=flat&amp;logo=github&amp;logoColor=white" alt="GitHub stars" /></a>
</p>

<p align="center">
  <a href="https://dlog.com.cn/learn/advanced-kernels/">Read the Blog</a>
</p>

## A Quick Note

Hi everyone, long time no see!

I’m Du Ziyuan, a Bilibili creator and tech blogger. I’d like to share what I’m learning about AI infrastructure, GPU programming, and ways to make large language models run faster.

I’m switching to English to reach more readers around the world. I also hope this series helps you feel more comfortable reading technical content in English. Let’s keep learning and sharing together!

Our earlier series covered the basics. This one moves into kernel fusion, performance analysis, and modern LLM workloads. Familiar operations such as RMSNorm, RoPE, Softmax, and SwiGLU become ingredients in larger optimization problems.

We'll use TileLang as the main implementation language, read other kernel ecosystems when they offer useful ideas, and follow one path throughout:

> **Math → Implementation → GPU execution → Profiling → Optimization**

Ready? Let's get started!

## Contents

- [A Quick Note](#a-quick-note)
- [Contents](#contents)
- [Quick Start](#quick-start)
- [Goals and Scope](#goals-and-scope)

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
