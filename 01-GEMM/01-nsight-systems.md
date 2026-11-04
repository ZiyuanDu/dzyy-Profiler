# Find the GEMM Performance Gap with Nsight Systems

In [the first lesson](00-basic-gemm.md), we checked correctness and measured TileLang against PyTorch. Suppose the benchmark consistently shows that TileLang takes longer. Our next question is **where that extra time occurs**: inside GPU kernels, around their launches, or in additional work.

This lesson follows one path:

> **Install → capture warmed calls → identify their kernels → inspect the timeline → choose the next investigation.**

We use Nsight Systems (`nsys`) to collect a report and its graphical application (`nsys-ui`) to explore it. The companion script is [`01_profile_gemm.py`](01_profile_gemm.py); it imports the same GEMM from the first lesson.

**Run project commands from the repository root.** Installation commands are the exception. You need the working project environment from [Quick Start](../README.md#quick-start) and an NVIDIA GPU accessible to that environment.

## 1. Know What This Capture Should Answer

Keep the shape that showed the gap. The commands below use `M = N = K = 1024`; replace all three values if your benchmark used another shape.

By the end, you should be able to answer:

1. Which GPU kernels belong to one TileLang call and one PyTorch call?
2. How long do those kernels execute on the GPU?
3. Are there additional kernels, copies, waits, or gaps around them?
4. Does the next investigation belong in the calling code or inside a particular GPU kernel?

`nsys` shows execution order and timing. Detailed questions about register pressure, shared-memory conflicts, or matrix-instruction efficiency lead to **Nsight Compute (`ncu`) after we identify the relevant kernel**. You can follow this lesson without adding `torch.profiler`.

## 2. Install the Collector and Report Viewer

### Check whether the CLI is already installed

On the GPU machine:

```bash
command -v nsys
nsys --version
nvidia-smi
```

`command -v` prints the selected executable's location. `nsys --version` identifies the profiler; `nvidia-smi` identifies the GPU and driver. A working CUDA or PyTorch installation does not guarantee that `nsys` is on your `PATH`.

If the first two commands work, keep that installation and continue to the capture section. Record its version with your results.

### Choose the right download

Open NVIDIA's [Nsight Systems download page](https://developer.nvidia.com/nsight-systems/get-started). Choose the package for the machine where each component will run:

| Your setup | On the GPU machine | Where you view the report |
| --- | --- | --- |
| Linux desktop | Linux **Full Version** | Its included Linux GUI |
| Linux server accessed through SSH | Linux **CLI Only** | Full Version on your desktop |
| WSL2 | Linux CLI inside WSL | Windows Full Version is a convenient viewer |

On Linux, check your distribution and CPU architecture before downloading:

```bash
cat /etc/os-release
uname -m
```

This repository targets Linux `x86_64`. For an Ubuntu machine with that architecture, select **Linux on x86_64**, then a `.deb` package. A Linux Arm server needs the corresponding Arm package. Match the selected release to NVIDIA's supported OS, GPU, CUDA, and driver requirements; use the download page's archives when needed.

For report viewing, choose the desktop's OS: Linux, Windows, or macOS. The GUI can view a report collected on a different machine. **Use the same version as the collector, or a newer viewer**, according to NVIDIA's [platform and report compatibility table](https://developer.nvidia.com/nsight-systems/get-started).

### Install a downloaded Ubuntu `.deb`

In the directory containing your download:

```bash
cd ~/Downloads
sudo apt install ./YOUR_DOWNLOADED_FILE.deb
```

Replace `YOUR_DOWNLOADED_FILE.deb` with the actual filename; shell Tab completion helps. The `./` tells `apt` to install a local file and resolve its dependencies. These commands install a package on your system; they are separate from `uv sync` and the project's Python environment.

Then verify:

```bash
command -v nsys
nsys --version
```

If you installed the Full Version, also check `command -v nsys-ui`. CLI-only installations provide the collector without the graphical viewer. Other package formats and installation methods are covered in NVIDIA's [installation guide](https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html).

<details>
<summary>Installed, but the shell cannot find nsys?</summary>

Check the installation location reported by the installer. CUDA Toolkit installations may also provide `/usr/local/cuda/bin/nsys`; this location depends on how CUDA was installed.

First run the executable by its full path. Once that works, add its containing directory to `PATH`:

```bash
export PATH="/absolute/path/to/directory-containing-nsys:$PATH"
```

Replace the example directory with your installation's real path. Add the resulting line to your shell configuration only if you want it in future terminals. `nsys` and `nsys-ui` may be in different directories.

</details>

## 3. Capture a Small, Clearly Labeled Region

Return to the repository root. Confirm that the companion script is available:

```bash
uv run --locked 01-GEMM/01_profile_gemm.py --help
```

The script prepares inputs, compiles the original kernel, checks its result, and warms up both implementations. **All of that finishes before collection starts.** NVTX supplies names for regions of code on the timeline; the script uses PyTorch's built-in NVTX functions, so no extra Python package is needed. Inside the capture, it runs 20 calls per implementation with these labels:

| Label | Meaning |
| --- | --- |
| `TileLang/gemm` | A CPU call launching the compiled TileLang GEMM |
| `PyTorch/gemm` | A CPU call to `torch.mm` |
| `cache_flush` | Clear a 256 MiB buffer before a GEMM call |
| `drain` | Wait for the current phase's queued GPU work to finish |

The outer `TileLang` and `PyTorch` ranges group each phase, including its preparation and final wait. **Their durations are not per-call GEMM latency.**

The default `--cache flush` uses the same 256 MiB int32-buffer clearing procedure as the first lesson's Event benchmark. Keep the flush visible so that you can distinguish benchmark preparation from GEMM execution. This reproduces the cache preparation, rather than the benchmark's entire CUDA Event sequence.

Collect the report:

```bash
mkdir -p profiles

nsys profile \
  --trace=cuda,nvtx,cublas \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --output=profiles/gemm_1024_flush \
  .venv/bin/python 01-GEMM/01_profile_gemm.py \
  --shape 1024 1024 1024 --iterations 20 --cache flush
```

Here is how to read that command:

| Part | Purpose |
| --- | --- |
| `nsys profile` | Launch the application under the collector |
| `--trace=cuda,nvtx,cublas` | Record CUDA activity, our labels, and cuBLAS API activity |
| `--sample=none --cpuctxsw=none` | Leave CPU sampling and thread-scheduling collection off for this first trace |
| `--capture-range=cudaProfilerApi` | Start at the script's CUDA profiler-start call |
| `--capture-range-end=stop` | Stop collection at its profiler-stop call; let the application finish |
| `--output=...` | Choose the report's location and filename prefix |
| `.venv/bin/python ...` | Run with the project's already-created Python environment |

CUDA API tracing remains enabled even though CPU sampling is off. `torch.cuda.profiler.profile()` in the script supplies the start/stop calls; it is a capture boundary for the external profiler. The corresponding collection controls are documented in NVIDIA's [focused profiling and CLI guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#focused-profiling).

Successful collection should create:

```text
profiles/gemm_1024_flush.nsys-rep
```

Keep this report. Statistics commands may additionally create a `.sqlite` file; `.nsys-rep` is the report to open or copy to your desktop. The repository already ignores the root-level `profiles/` directory.

If the output filename already exists, choose another prefix, such as `gemm_1024_flush_run2`, and use that name in subsequent commands. The profile may take several seconds to generate even though the captured GPU work is short.

## 4. Identify the Work from the Terminal

First, associate GPU kernels with our labels:

```bash
nsys stats --report nvtx_kern_sum profiles/gemm_1024_flush.nsys-rep
```

Look specifically at rows whose **NVTX Range** is `TileLang/gemm` or `PyTorch/gemm`. Read these fields together:

| Field | What to inspect |
| --- | --- |
| `Kernel Name` | The actual GPU function that executed |
| `NVTX Inst` | Number of matching labeled call ranges |
| `Kern Inst` | Number of kernel executions in that row |
| `Med` / `Avg` | Per-kernel duration, with the units printed in the report |
| `Total Time` | Accumulated duration for that row's kernel executions |

There are 20 labeled calls per implementation. One PyTorch call can launch multiple kernels, so inspect **all** rows for `PyTorch/gemm`. Different shapes can select different library kernels.

With nested ranges, one kernel can appear under both an inner range and its parent. **Use the exact `/gemm` ranges for comparison; adding parent and child rows would double-count work.** If you sum their kernel durations and divide by 20, you obtain average accumulated GPU kernel time per logical call. That quantity excludes gaps and is not elapsed latency when kernels overlap.

For CUDA API activity:

```bash
nsys stats --report cuda_api_sum profiles/gemm_1024_flush.nsys-rep
```

Use it to spot launches, waits, or unexpected allocation/copy APIs. A long synchronization call often waits for work already running on the GPU; its full duration cannot automatically be counted as extra overhead. The script deliberately synchronizes at phase boundaries.

For individual launches, including their grid and block dimensions:

```bash
nsys stats --report cuda_kern_exec_trace:nvtx-name profiles/gemm_1024_flush.nsys-rep
```

The prefix identifies the innermost NVTX range. `API Dur` describes the launch API on the CPU, `Queue Dur` describes waiting before GPU execution, and `Kernel Dur` describes GPU execution. These intervals can overlap. A long queue can simply mean the GPU is busy with earlier work, including our cache flushes.

Report availability and columns vary by version. Use `nsys stats --help-reports` to list yours and `nsys stats --help-reports nvtx_kern_sum` for its field definitions. NVIDIA's [post-collection analysis guide](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html#statistical-analysis) documents these reports. For a general kernel summary, `cuda_gpu_kern_sum` is also useful: its `Time (%)` is a share of accumulated listed kernel time, not GPU utilization or a share of application wall time.

## 5. Open the Timeline and Follow One Call

### Open the report

On a Linux desktop with the Full Version:

```bash
nsys-ui profiles/gemm_1024_flush.nsys-rep
```

Alternatively, launch NVIDIA Nsight Systems from your application menu and use **File → Open**.

If collection happened on a server, run the following on your **desktop**, replacing the SSH destination and absolute report path:

```bash
scp USER@SERVER:/absolute/path/to/profiles/gemm_1024_flush.nsys-rep .
```

Open the copied file in the desktop GUI. With WSL2, a Windows Nsight Systems GUI can open a copied report through **File → Open**. You do not need to install the project or run its Python code on the viewing machine.

### Read the three relevant rows

Expand the process/thread and GPU rows in **Timeline View**. Names and nesting vary by version, but locate:

| Row | What its rectangles represent |
| --- | --- |
| CPU thread → NVTX | Our named Python scopes |
| CPU thread → CUDA API | CPU runtime/driver calls that submit or synchronize GPU work |
| GPU → CUDA context → stream | Actual kernel and memory-operation execution on the GPU |

Start with a middle call in the `TileLang` phase, then one in the `PyTorch` phase. Zoom until individual events are readable. Select a launch API and inspect the correlated GPU event; hover or select that GPU event to read its name, duration, stream, grid, and block details where available. Follow the correlation instead of identifying ownership only by horizontal proximity.

**An NVTX range is a CPU label.** Python can return from a launch before the GPU finishes. Consequently, a `/gemm` NVTX rectangle can be shorter than its GPU kernel. Read GPU duration from the GPU event or kernel report. NVIDIA explains this relationship in its [NVTX and CUDA tracing guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#marking-and-labeling-regions).

For each implementation, inspect several calls and note:

1. The kernel or kernels launched by a single `/gemm` call.
2. Their GPU durations and whether those durations are consistent.
3. Copies, allocations, or synchronization inside that call, if any.
4. GPU work immediately before the call and any idle interval before it starts.

The default TileLang output grid is `8 × 8` blocks with 128 threads per block. Confirm the report's launch information rather than assuming a generated kernel name. Grid size alone does not establish a bottleneck.

## 6. Decide What the Evidence Supports

Use this table after identifying which events actually belong to each call:

| Observation | Interpretation to investigate | Next step |
| --- | --- | --- |
| TileLang's GEMM GPU execution is consistently longer | The kernel implementation contributes to the gap | Save its name and launch details; use `ncu` to investigate hardware execution |
| GPU execution is similar, but repeated calls leave different idle gaps | Host submission or dependencies may contribute | Inspect correlated APIs and a no-flush trace before changing GEMM tiling |
| One implementation launches extra kernels or copies | Additional work contributes | Determine each operation's role and compare the full logical call |
| Most captured time belongs to `cache_flush` | The trace includes the intended cache preparation | Compare `/gemm` events separately; keep cache policy recorded |
| Only the first captured call is unusually slow | A one-time effect may remain | Check warmup and repeated captures before explaining steady-state performance |

Both kernel execution and surrounding work can contribute. Record them separately and use the timeline to see how they combine.

Three observations need particular care:

- **An empty stream row is not proof that the entire GPU is idle.** Inspect the other relevant streams; this capture also does not describe every other process using the device.
- **Queue time is not idle time.** Look at what the GPU was doing while the kernel waited.
- **Our `cache_flush` and `drain` scopes are harness operations.** Keep them distinct from work launched inside `/gemm`.

If GPU execution is longer, this trace has narrowed the investigation. It has not yet shown whether the cause is insufficient parallelism, data movement, synchronization, or instruction throughput. Those are questions for the selected kernel's `ncu` report.

## 7. Optional: Inspect Repeated Calls Without Cache Flushes

The flush trace helps interpret the cache conditions used by the original benchmark. Its clearing operations also keep the GPU occupied between GEMMs, which can hide differences in how quickly the CPU submits repeated calls.

To inspect submission and gaps under a second, explicitly different cache condition:

```bash
nsys profile \
  --trace=cuda,nvtx,cublas \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --output=profiles/gemm_1024_warm \
  .venv/bin/python 01-GEMM/01_profile_gemm.py \
  --shape 1024 1024 1024 --iterations 20 --cache warm
```

`--cache warm` reuses the same tensors without clearing the buffer between calls. It allows cache reuse; it does not guarantee that every input remains cached. Compare TileLang and PyTorch **within this report**, then explain differences from the flush report as a separate experiment. It is not a replacement measurement for the first lesson's flush-before-call benchmark.

NVTX and tracing add overhead, especially around short calls. This first collection also leaves CPU sampling and scheduling traces off, so it cannot explain every source of a host delay. If gaps become the main finding, a focused follow-up host trace can add that information. Final speed claims should still come from the independent benchmark.

## 8. Troubleshoot the First Capture

| Symptom | What to check |
| --- | --- |
| `nsys: command not found` | Installation and `PATH`; try the installed executable's full path |
| `nsys-ui` is missing on the server | A CLI-only installation is sufficient there; open the report on a desktop with the Full Version |
| Permission warnings mention CPU sampling or context switches | Confirm both `--sample=none` and `--cpuctxsw=none`; this lesson does not require those collections |
| No report, or no CUDA kernel events | Check the application exit status and diagnostic messages; confirm the working benchmark and that this script reached its capture region |
| CUDA tracing is unavailable | Check the selected Nsight release's driver/GPU/CUDA support; avoid collecting with another CUDA profiler at the same time |
| The report opens as unsupported | Check the viewer version against the collector version |
| Compilation or tensor initialization dominates the timeline | Confirm that you used the companion script with `--capture-range=cudaProfilerApi` |
| An output file already exists | Choose a new `--output` prefix |

Useful diagnostic commands are `nsys --version`, `nsys profile --help`, and `nsys status --environment`. NVIDIA's [release notes](https://docs.nvidia.com/nsight-systems/ReleaseNotes/index.html) cover version-specific limitations; the project's [troubleshooting index](../troubleshooting/README.md) covers environment and compilation problems.

## 9. Write Down the Finding Before Optimizing

Fill this small record from your own report:

```text
GPU / driver / nsys version:
Shape / dtype / precision settings:
Independent benchmark: TileLang ___ us; PyTorch ___ us
Capture: flush or warm; 20 calls per implementation
TileLang kernel name(s), count per call, and GPU duration(s):
PyTorch kernel name(s), count per call, and GPU duration(s):
Additional work or idle gaps, with timeline evidence:
Conclusion supported by this capture:
Next question and tool:
```

For example, a finding might say: “Under the same flush-before-call condition, TileLang's GEMM GPU execution is consistently longer. The next step is an `ncu` comparison of the identified kernels.” It should include your measured durations and report name before being treated as an experimental result.

The commands and report names in this lesson were checked against installed Nsight Systems **2025.6.3** help and NVIDIA's online documentation on **November 4, 2026**. No GPU capture or measured bottleneck is supplied with this lesson; use the steps above to produce evidence on your own machine.
