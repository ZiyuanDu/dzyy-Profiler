# CUDA compilation fails

If TileLang fails with `Value 'sm_120a' is not defined for option 'gpu-architecture'`, check the CUDA compiler it invokes.

## Symptoms

Running the basic GEMM example:

```bash
uv run --locked 01-GEMM/00_basic_gemm.py
```

TileLang starts compilation, then prints the generated CUDA source followed by:

```text
Compilation error:
nvcc fatal   : Value 'sm_120a' is not defined for option 'gpu-architecture'

Command: /usr/bin/nvcc ... --cubin ... -arch=sm_120a ...
```

The environment where this error was observed:

| Component | Observed value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090 |
| NVIDIA driver | 595.91.07 |
| `CUDA Version` reported by `nvidia-smi` | 13.2 |
| Compiler invoked by TileLang | `/usr/bin/nvcc`, CUDA 12.0 |
| `CUDA_HOME` | Unset |

## Root cause

**TileLang targets `sm_120a` for the RTX 5090, but the selected CUDA 12.0 compiler does not recognize that target.** Compilation fails while parsing the architecture option, before the GEMM kernel runs.

`sm_120a` is a Blackwell architecture target supported by CUDA Toolkit 12.8. This project's PyTorch build uses `cu130`, so the resolution below uses CUDA Toolkit 13.0. See the [NVIDIA compiler documentation](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-compiler-driver-nvcc/index.html) and the [project dependencies](../pyproject.toml).


## Diagnosis

Run these commands from the project root:

```bash
which nvcc
nvcc --version
echo "$CUDA_HOME"
nvidia-smi

# Print the compiler path TileLang will actually use.
uv run --locked python -c \
  'from tilelang.contrib.nvcc import get_nvcc_compiler; print(get_nvcc_compiler())'
```

Compare the compiler path printed by TileLang with the one returned by `which nvcc`. If they differ, check whether `CUDA_HOME` still points to an older Toolkit.

## Resolution

Choose one of the following options.


### Install compiler components in the project environment

If a suitable Toolkit is unavailable, install NVIDIA's CUDA 13.0 compiler and C++ headers in the project's virtual environment. This option does not require `sudo`. See the [CUDA Toolkit package documentation](https://pypi.org/project/cuda-toolkit/13.0.3/).

```bash
# Add compiler components and update pyproject.toml, uv.lock, and .venv.
uv add "cuda-toolkit[nvcc,cccl]==13.0.3.0"

# Locate the installed compiler and select it explicitly.
export CUDA_HOME="$(uv run --locked python -c '
from importlib.metadata import files
print(next(f.locate().parent.parent for f in files("nvidia-cuda-nvcc") if f.name == "nvcc"))
')"
export PATH="$CUDA_HOME/bin:$PATH"
```

Setting `CUDA_HOME` matters even after installing these packages. TileLang 0.1.13 checks environment variables and the `nvcc` on `PATH` before looking for the Python package. Without an explicit selection, it can continue to find the old `/usr/bin/nvcc`. See [TileLang's CUDA discovery logic](https://github.com/tile-ai/tilelang/blob/v0.1.13/tilelang/env.py).



### Use an existing CUDA Toolkit installation

If CUDA Toolkit 13.0 is already installed, select it explicitly. Replace the example path with the actual installation directory. On a cluster, you can first load the CUDA 13.0 module provided by the administrator.

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
```

## Verification

Run the following commands in the same terminal:

```bash
nvcc --version
uv run --locked 01-GEMM/00_basic_gemm.py
```

