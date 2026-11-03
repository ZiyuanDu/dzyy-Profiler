# Set Up

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


Use `uv run` for project commands.

```bash
uv run --locked 01-GEMM/00_basic_gemm.py
```
Look for **`correctness passed`**, followed by TileLang and PyTorch timings. The first run includes JIT compilation; later runs can reuse the cache.

Then try the default workload and an irregular shape:


