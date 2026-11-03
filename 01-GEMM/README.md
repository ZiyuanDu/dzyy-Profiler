# 矩阵乘法：从大模型组件到 Profiling 驱动的优化

这篇教程以 TileLang 为实现主线，把矩阵乘法与 Linear、Attention 和 MoE 联系起来。每次优化都要形成一条可以验证的证据链：

> 工作负载与参数 → 测量结果 → 瓶颈假设 → 实现修改 → 再次测量 → 保留或推翻假设。

例如，“增加流水线 stage”只是一次修改；“原实现存在加载依赖，增加 stage 后计算与加载重叠改善，资源占用仍可接受，最终延迟下降”才是需要通过实验建立的结论。

本文提供教程组织、参考仓库、分析方法与实验模板。第一课现已提供[最小 GEMM 示例及讲解](00-basic-gemm.md)，可以从项目根目录用 `uv run 01-GEMM/00_basic_gemm.py` 运行；后文 `v00_simt` 至 `v07_dispatch` 等版本仍是实现规划。外部仓库命令需要在对应仓库中运行，引用的历史结果、公式推导和假设数据分别标明。

## 目录

- [先跑通第一课：TileLang 基础与最小 GEMM](00-basic-gemm.md)
- [1. 参考哪些 GitHub 仓库](#1-参考哪些-github-仓库)
- [2. 四个模块就够了](#2-四个模块就够了)
- [3. 先把矩阵乘映射到模型组件](#3-先把矩阵乘映射到模型组件)
- [4. 建立统一的测量方法](#4-建立统一的测量方法)
- [5. 模块一：稠密 GEMM 与形状变化](#5-模块一稠密-gemm-与形状变化)
- [6. 如何读指标并决定下一步](#6-如何读指标并决定下一步)
- [7. 三个分析案例](#7-三个分析案例)
- [8. 模块二至四：连接 Attention、MoE 与稀疏计算](#8-模块二至四连接-attentionmoe-与稀疏计算)
- [9. Profiling 命令与报告保存](#9-profiling-命令与报告保存)
- [10. 每次实验应该交付什么](#10-每次实验应该交付什么)

## 1. 参考哪些 GitHub 仓库

核查日期：2026-11-02。以下判断来自仓库、作者文章与官方文档的阅读，未在本机编译复现这些项目。判断教学价值时，优先看是否解释修改原因、提供可核对的测量方法、说明结果适用的硬件和形状。

| 仓库 | 适合借鉴的内容 | Profiling 与证据 | 使用时的边界 |
| --- | --- | --- | --- |
| [siboehm/SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA) | 从朴素实现到合并访存、共享内存、寄存器分块、warp tiling 的演进 | [作者优化日志](https://siboehm.com/articles/22/CUDA-MMM)结合访存吞吐、warp 状态、资源占用解释优化；仓库提供 `make profile KERNEL=...` | 优先推荐作为讲解范本；主线是 FP32 SIMT SGEMM，不能把其性能数值直接套到 BF16 Tensor Core GEMM；部分版本对形状有要求 |
| [leimao/CUDA-GEMM-Optimization](https://github.com/leimao/CUDA-GEMM-Optimization) | 对照多个 CUDA 实现，理解分块、向量化、寄存器复用以及 Tensor Core 扩展 | [配套长文](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/)详细推导实现，仓库提供计时比较与 Nsight 使用说明 | 适合补充实现细节；教程需要自行补齐每一版的计数器报告和瓶颈判断；仓库参数针对特定 GPU 与形状调过 |
| [open-infra-ai/cuda-foundations](https://github.com/open-infra-ai/cuda-foundations) | SGEMM 演进、benchmark 组织及中英文学习路径 | 有独立的 [Nsight 操作指南](https://open-infra-ai.github.io/cuda-foundations/en/guides/profiling)和[实测结果说明](https://open-infra-ai.github.io/cuda-foundations/en/benchmarks/) | 可借鉴实验组织；其 profiling 页面注明当前环境缺少完整真实 NCU 指标演示。部分指标到原因的简化对应，仍应按 NVIDIA 文档复核 |
| [csl-iisc/gpu-profiling-tutorial](https://github.com/csl-iisc/gpu-profiling-tutorial) | 用小型 GEMM 学习 nsys、ncu、NVTX 与报告查看 | 仓库包含采集脚本和预生成的 `.ncu-rep`、`.nsys-rep` | 适合作为工具练习；H2D/D2H、pinned memory 的实验需要与纯设备 GEMM 性能分开解释 |
| [tile-ai/tilelang](https://github.com/tile-ai/tilelang) | 本系列的实现依据；从[官方 GEMM 示例](https://github.com/tile-ai/tilelang/blob/main/examples/gemm/example_gemm.py)理解 shared、fragment、`T.gemm`、`T.Pipelined` | 示例包含正确性与计时流程，便于建立可用基线 | 官方示例已经组合了多种优化；教学中应拆开做对照，并检查编译后实际发生的变化 |
| [triton-lang/triton](https://github.com/triton-lang/triton) | [官方矩阵乘教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)中的分块、block 顺序与自动调优 | 明确解释 L2 复用的动机，并提供配置搜索与 benchmark | 适合跨 DSL 对照；单独一张 benchmark 曲线仍不足以证明缓存命中率是加速原因 |

建议按“Simon Boehm 的分析方式 → Lei Mao 的实现细节 → TileLang 官方实现”阅读，用 Nsight 官方文档校对指标含义。cuda-foundations 和 profiling-tutorial 则补充测量流程与报告组织。

这些资源各有侧重。本教程需要补充的是：同一批模型形状上的连续实验、原始报告，以及每次修改前后的因果解释。

## 2. 四个模块就够了

分类有多个可组合的维度：稠密/稀疏描述数据结构，GEMV/大 GEMM 描述形状，Batched/Grouped 描述任务组织，BF16/INT4 描述数值表示。

| 教学模块 | 模型组件 | 核心问题 | 第一版的实现范围 |
| --- | --- | --- | --- |
| 一：稠密 GEMM 与形状变化 | QKV/输出投影、Dense FFN、LM Head | 数据如何复用？prefill 与 decode 为什么需要不同配置？ | 一条完整的优化线，包含大 M、小 M 与 GEMV 对照 |
| 二：Batched GEMM | Attention 的 QKᵀ、PV | 多个矩阵乘如何组织？中间张量是否必须写回？ | 规则 batch/head 的 reference，连接后续 FlashAttention 章节 |
| 三：Grouped GEMM | MoE 专家 | 每个专家的 token 数不同，如何减少 padding 和启动开销？ | 逐专家、padding、grouped 三种执行方式对照 |
| 四：稀疏矩阵乘 | 块稀疏 Attention、稀疏权重 | 跳过计算的收益，能否覆盖索引与调度成本？ | 实现一种块稀疏×稠密；概览非结构化与 2:4 稀疏 |

模块一占主要篇幅，其余模块复用分块、布局、流水线与测量方法。低精度放在扩展实验中，先统一 BF16/FP16 输入、FP32 累加；SIMT 原理演示可以单独使用 FP32。

## 3. 先把矩阵乘映射到模型组件

统一数学约定：

$$
C_{M\times N}=A_{M\times K}W_{K\times N}
$$

`M` 表示输入行数，`K` 表示归约/输入通道维度，`N` 表示输出通道维度。权重按数学方向书写；实际框架可能存储 `[N,K]` 并进行转置访问，benchmark 必须记录布局与 stride。

记 `T` 为本次参与线性层计算的 token 数，`D` 为隐藏维度，`F` 为 FFN 中间维度，`V` 为词表大小。`Hq/Hkv` 为 query/KV 头数，`d` 为头维度，`Lq/Lkv` 为 query/KV 长度。

| 组件 | 矩阵形状 | 特点 |
| --- | --- | --- |
| Q 投影 | `[T,D] × [D,Hq*d]` | 不同 token 共享权重 |
| K、V 投影 | `[T,D] × [D,Hkv*d]` | GQA 下输出宽度可小于 Q 投影 |
| Attention 输出投影 | `[T,Hq*d] × [Hq*d,D]` | 稠密 GEMM |
| SwiGLU gate/up | 两次 `[T,D] × [D,F]` | 后续接 SiLU 与逐元素乘 |
| FFN down | `[T,F] × [F,D]` | 输入、输出通道与 up 投影互换 |
| QKᵀ | 每个 query head：`[Lq,d] × [d,Lkv]` | 样本/head 维度组织成多个任务 |
| PV | 每个 query head：`[Lq,Lkv] × [Lkv,d]` | P 是 Softmax 后的注意力权重 |
| MoE 专家 up/gate | 每个专家：`[Te,D] × [D,F]` | 各专家的 `Te` 动态变化 |
| LM Head | `[T,D] × [D,V]` | N 可能很大；推理时需说明是否只计算最后位置 |

上述形状对应常见 decoder-only Transformer，可对照 [Llama 官方实现](https://github.com/meta-llama/llama/blob/main/llama/model.py)。GQA 需要正确建立 query head 与 KV head 的映射；实际实现不必复制 K/V 到每个 query head。[PyTorch SDPA 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

第一版专注前向、prefill 和 decode。反向只需说明：对 `Y=XW`，有 `dX=dY·Wᵀ`、`dW=Xᵀ·dY`，仍可复用 GEMM 知识，但形状和布局发生变化。

## 4. 建立统一的测量方法

### 4.1 三种工具回答三个层次的问题

| 工具 | 回答的问题 | 首先查看什么 |
| --- | --- | --- |
| CUDA Events / 可靠的设备 benchmark | 修改后是否更快？ | 稳态延迟分布、相对基线收益 |
| Nsight Systems (`nsys`) | 时间花在哪一段？ | CPU 发射、kernel 间隔、拷贝、同步、各 kernel 耗时 |
| Nsight Compute (`ncu`) | 选中的 kernel 被什么限制？ | 网格与资源、内存层次、计算管线、调度与依赖 |

先定位值得优化的区间，再分析其中的 kernel。`nsys` 的 kernel 汇总 `Time (%)` 通常表示该项在汇总 kernel 累计时间中的占比，不等于整个程序墙钟时间占比或 GPU 利用率；存在并发时还要回到时间线判断。[Nsight Systems 用户指南](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)

### 4.2 计时约定

1. 输入常驻 GPU；分配、随机初始化、首次编译和正确性检查放在计时区间外。需要研究数据传输时，另报端到端结果。
2. 在目标 shape/配置上预热，计时前明确 stream 依赖。使用 CUDA Events 测同一 stream 的设备工作，避免只用 CPU 计时器包住一次异步调用。
3. 重复采样，记录中位数与波动，例如 p10/p90；声明一次采样包含多少次调用，以及延迟是逐次事件测量还是整批时间除以调用次数。多 stream 工作需建立完整依赖。
4. 相同输入类型、累加类型、输出类型、布局和数学操作之间比较。FP32 基线注明是否允许 TF32；Tensor Core 与 SIMT 使用不同精度时，另报误差。
5. 正确性检查覆盖随机输入、边界形状与数值误差。根据 dtype、K 和数据范围确定容差，避免只检查全 1 输入的一个输出元素。
6. 单独记录普通发射与 CUDA Graph 模式；测量不同形状时重新预热，保持 GPU 并发负载、频率与功耗条件尽量一致。
7. 声明重复使用同一组地址，还是轮换多个输入/权重。权重可驻留 L2 的热缓存实验，与接近模型逐层读取权重的实验分开报告。

`ncu` 可能多次 replay kernel，且默认缓存处理会改变访存状态。因此延迟收益以统一的独立 benchmark 为准，NCU 用于解释；比较报告时保持 replay/cache 设置一致。研究真实热缓存行为时，可评估 application replay 配合 `--cache-control none`。[NCU 缓存控制说明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#cache-control)

### 4.3 固定一小组有模型含义的 shape

以下是教学工作负载，并非某个完整模型的所有真实配置；显存较小时可以统一缩小通道维度。

| case | M | N | K | 希望观察什么 |
| --- | ---: | ---: | ---: | --- |
| 小规模原理演示 | 128 | 128 | 128 | 线程映射与正确性；限制朴素实现的运行时间 |
| Prefill 投影 | 2048 | 4096 | 4096 | 权重复用、Tensor Core 与流水线 |
| Decode 单 token | 1 | 4096 | 4096 | GEMV、权重流量与有限并行度 |
| Decode 多请求 | 32 | 4096 | 4096 | 小 M GEMM、batch 增大后的权重复用 |
| FFN up | 512 | 11008 | 4096 | 宽输出与 tile 选择 |
| 非整除边界 | 129 | 257 | 259 | mask、尾块和额外工作 |

日常调优只深入分析其中 1–2 个 case，选出的配置再回归整组形状。完整 FFN 评估还应加入 down 投影。

## 5. 模块一：稠密 GEMM 与形状变化

### 5.1 先推导三个量

对 `C=AW`，忽略后处理，乘加按两次浮点运算计算：

$$
F=2MNK,\qquad P_{\mathrm{measured}}=\frac{2MNK}{t}
$$

若 A/W/C 的元素字节数分别为 `sA/sW/sC`，定义无额外重读、无预先缓存时的理想张量流量：

$$
Q_{\mathrm{ideal}}=s_A MK+s_W KN+s_C MN,
\qquad AI_{\mathrm{ideal}}=F/Q_{\mathrm{ideal}}
$$

它是分析基准，不是实际 HBM 流量。缓存命中可使测得的 HBM 流量更小，重读、事务浪费和 spill 可使其更大；`βC`、量化元数据与融合操作还要加入相应读写。

固定 `N=K=4096`，假设三个张量均为两字节元素，可推导：

| M | 计算量 | 理想张量流量 | 理想算术强度 |
| ---: | ---: | ---: | ---: |
| 1 | 33.55 MFLOP | 32.016 MiB | 约 1.00 FLOP/byte |
| 32 | 1.074 GFLOP | 32.5 MiB | 约 31.51 FLOP/byte |
| 2048 | 68.72 GFLOP | 64 MiB | 1024 FLOP/byte |

这是公式计算，说明 token 数会改变权重复用。若考虑某一级内存的实际流量 `Q` 与该层带宽 `BW`，可用 `max(F/P_peak, Q/BW)` 估计理想时间下界；计算峰值必须匹配 dtype、指令路径和稠密/稀疏模式。实际 kernel 还会受启动、依赖和资源限制。基础背景见 [NVIDIA GEMM 性能指南](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)。

### 5.2 版本演进与对应的证据

| 版本规划 | 主要修改 | 修改前需要找的证据 | 修改后如何验证 |
| --- | --- | --- | --- |
| `v00_simt` | 每个线程直接计算输出 | 先获得正确性、时间、生成代码与访存映射 | 小 shape 对齐 reference，记录基线 |
| `v01_shared` | 合并加载，共享内存分块 | global load 的事务效率、重复读取和缓存层流量 | 对照请求/sector、L1/L2/DRAM 流量与时间；允许瓶颈转到 shared/计算 |
| `v02_tensorcore` | 使用 `T.gemm` 与合适的累加类型 | 原实现指令路径、精度要求与矩阵形状 | 检查矩阵乘指令、对应计算管线、误差与时间 |
| `v03_pipeline` | 使用 `T.Pipelined`，扫描 stage | eligible warp/issue 活跃度、加载依赖、是否存在重叠空间 | 同时比较等待、计算管线、shared/寄存器占用与时间 |
| `v04_tuned_tiles` | 调整 BM/BN/BK、线程数、stage | tile 数、尾块、资源限制和每个线程的工作量 | 有限搜索；分别记录大 M 与小 M 的最佳配置 |
| `v05_layout` | 调整布局或 block 顺序 | 源码位置对应的 excessive wavefront、事务浪费或 L2 重读 | 检查生成代码是否变化，再看目标计数器与时间 |
| `v06_epilogue` | 融合 bias/激活等后处理 | 时间线中间 kernel、额外张量读写 | 比较相同完整操作的总时间，检查寄存器压力 |
| `v07_dispatch` | 根据 shape 选择少量配置 | 固定配置在不同 case 上的退化 | 回归整组 workload，记录选择规则和调优成本 |

为分清合并访存与 shared tiling 的贡献，可以在 `v01_shared` 内保留两个小对照。SIMT 部分的寄存器分块只做必要解释，主要性能路线尽早转到 Tensor Core。

官方 TileLang 示例已包含多个优化。拆解时需要注意：源码上移除一项设置，不保证编译器就生成完全不同的代码；增加 layout 设置也可能得到同样的代码。[官方 GEMM 示例](https://github.com/tile-ai/tilelang/blob/main/examples/gemm/example_gemm.py)、[软件流水线文档](https://www.tilelang.com/programming_guides/software_pipeline.html)

### 5.3 参数可以统计出哪些约束

对常规、未做 split-K 的单个 GEMM：

$$
N_{\mathrm{CTA}}=\lceil M/B_M\rceil\lceil N/B_N\rceil
$$

先比较 CTA 数与 GPU 的 SM 数，再查看每个 SM 的可驻留 block 数和 waves。单个输出 tile 不足以提供足够并行任务时，更大的 tile 可能让 GPU 更空。

对使用显式多缓冲的普通加载流水线，可先估计：

$$
S_{\mathrm{buffers}}\approx stages\cdot(s_A B_M B_K+s_W B_K B_N)
$$

实际还包含 padding、同步对象或编译器对缓冲的复用，应以 Launch Statistics 和生成代码为准。不能用这个估算替代真实 shared memory 占用。

以 `BM=BN=128, BK=32`、两字节输入为例，每份 A/B 缓冲合计 16 KiB；2/3/4 份缓冲对应约 32/48/64 KiB。增加缓冲可能改善重叠，也可能跨过驻留 block 的资源阈值。

更大的输出 tile 通常还意味着更多累加器。寄存器实际分配由编译器、线程分工与 MMA 路径共同决定，应检查 Registers Per Thread、occupancy 限制以及 spill/local 访问。

## 6. 如何读指标并决定下一步

### 6.1 先看问题，再选计数器

以下名字是常见 Nsight Compute 指标或 section 字段。硬件、NCU 版本和复制指令路径会影响可用性；先用 `ncu --list-sections`、`ncu --query-metrics` 查询。LSU global-load 指标不能完整覆盖所有异步复制/TMA 路径。

| 问题 | 首先查看 | 能支持的判断 |
| --- | --- | --- |
| 修改是否改善性能？ | 独立 benchmark；NCU 中 `gpu__time_duration.sum` 辅助核对 | 稳态延迟是否下降，波动是否覆盖收益 |
| 实际访问了多少显存？ | `dram__bytes_read.sum`、`dram__bytes_write.sum` | 与算法流量、L2 流量和缓存策略对照 |
| DRAM 是否接近吞吐上限？ | `dram__throughput.avg.pct_of_peak_sustained_elapsed` | 配合实际 bytes、时间和内存层次判断带宽压力 |
| 哪条计算管线忙？ | Compute Workload Analysis；`sm__throughput.avg.pct_of_peak_sustained_elapsed` 作入口 | 展开 Tensor/FP32/整数/访存等管线，避免把 SM 总览当作 Tensor Core 效率 |
| 驻留任务受什么限制？ | Occupancy、Launch Statistics 的寄存器/shared/block/网格信息 | 区分资源受限、网格过小与执行尾部不均衡 |
| 驻留 warp 能否发射？ | Scheduler Statistics 的 Active/Eligible Warps、Issue Active | 区分“没有任务”与“任务在等依赖” |
| global load 是否浪费事务？ | Memory Workload Analysis 的 Sectors/Req、Excessive Sectors | 结合访问宽度、对齐、活跃线程和源指令检查 |
| shared memory 是否串行化？ | Shared 表的 excessive wavefront、bank conflicts 与源指令 | 定位具体布局/访问模式；聚合 bank-conflict 值还可能包含其他冲突来源 |
| 是否存在 spill 或同步等待？ | Local memory 访问、生成代码、Warp State Statistics/Source Counters | 定位临时变量、依赖与 barrier；local 访问也可能来自显式局部数组 |

指标定义与诊断流程分别参考 [NCU Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) 和 [Compute Triage Guide](https://docs.nvidia.com/nsight-compute/ComputeTriage/index.html)。每次只采集回答当前问题所需的 sections，避免一开始就对所有配置采集 full report。

### 6.2 用“证据组合”提出假设

| 观察到的组合 | 下一步假设 | 可以尝试的修改 | 必须再检查什么 |
| --- | --- | --- | --- |
| 网格很小，计算和内存吞吐都低 | 并行任务不足 | 减小输出 tile；合并小任务；必要时研究 split-K | 额外归约/启动成本；多 batch 是否仍受益 |
| DRAM 流量大且吞吐接近可持续带宽 | 流量限制明显 | 提高复用、减少中间张量；满足精度要求时压缩权重 | bytes 与总延迟是否同时改善 |
| DRAM 吞吐低、eligible warp 少、加载相关依赖突出，其他层也未饱和 | 在途访存不足或依赖暴露 | 流水线、预取、更多独立工作 | 寄存器/shared 占用，以及等待是否转移 |
| 请求的 sector 明显多于相同活跃线程下的理想值 | 访存事务浪费 | 调整线程映射、对齐和访问布局 | 内存层次流量与时间；广播访问不要误判 |
| shared excessive wavefront 较高，相关源指令耗时突出 | shared 访问可能串行化 | 改布局、swizzle，必要时 padding | MMA/复制布局兼容性，以及新布局资源成本 |
| 寄存器增多、出现 spill，时间退化 | 每线程工作或变量存活范围过大 | 减小 tile、减少展开或临时值 | local 流量、发射效率与延迟；限制寄存器也可能增加 spill |
| Tensor pipe 已较忙、grid 足够、访存无明显问题 | 计算路径可能接近当前配置上限 | 减少指令/尾块；选择更适配的形状配置 | 相同精度的基线，以及完整组件收益 |
| 时间线上短 kernel 很多，区间总时间明显大于关键计算时间 | 发射、同步或中间步骤成本明显 | 融合、批量组织，按场景评估 CUDA Graph | 整个区间时间和融合后的资源占用 |

这些对应关系是待验证的假设。不要固定“低于某个 occupancy 百分比就必须优化”，也不要仅凭 Long Scoreboard 宣判 DRAM 带宽不足：它反映 L1TEX 路径相关依赖，来源可能涉及不同内存层与访问效率。`Not Selected` 表示有可发射 warp 但调度器选择了其他 warp，通常不是需要消除的等待。

### 6.3 派生指标的使用边界

- `TFLOP/s = 2MNK / 秒 / 10^12`：报告有效数学工作；mask、padding 的额外硬件工作单独解释。
- `实测 DRAM 带宽 = (DRAM read bytes + write bytes) / kernel 秒数`：使用同一采集口径下的时间和 bytes，注明缓存/replay 设置。
- `算法有效带宽 = Q_ideal / 独立 benchmark 秒数`：用于工作量比较，名称中保留“算法有效”，不能当作真实 HBM 带宽。
- `稀疏有效吞吐`：区分按实际非零乘加计算的吞吐，与按完整稠密工作量折算的 dense-equivalent 吞吐。

性能收益最终看延迟和目标组件时间。缓存命中率、occupancy、stall 百分比都不能单独作为验收标准。

## 7. 三个分析案例

### 7.1 合并访存：从线程地址分布到事务效率

先做地址推导：32 个活跃线程各加载一个互不重叠的 FP32 元素，连续且按 32-byte sector 对齐时，共请求 128 bytes，对应 4 个 sector。若每个线程访问不同 sector，则可能需要 32 个 sector。

因此，对相同源指令和有效工作，观察 Sectors/Req 是否接近该访问模式的预期值，再检查跨层流量和耗时。4 不是所有指令的通用目标：向量宽度、广播、活跃线程数和对齐都会改变预期。[NCU 内存表说明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#memory-tables)、[CUDA 合并访存说明](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)

真实历史例子：Simon Boehm 在其 A6000 上的 FP32 SGEMM 日志中报告，调整线程到输出元素的映射后，GMEM throughput 约从 15 GB/s 增加到 110 GB/s，计算吞吐约从 300 增加到 2000 GFLOP/s。这是作者特定实验中的近似结果，不能用来预估本机 TileLang 的加速倍数。[原始分析](https://siboehm.com/articles/22/CUDA-MMM)

课堂复现需要保存原始报告。如果只有延迟变好，应表述为“该映射更快；地址推导支持合并访存假设”，并继续采集事务证据。

### 7.2 从 shared 到寄存器：瓶颈可以转移

Simon Boehm 的 shared-memory 版本通过数据复用降低 global 访问后，继续结合 warp 状态中的 MIO Throttle 和内层 shared 访问分析，提出每个线程计算多个输出、在寄存器中复用数据的改进。[作者对 Kernel 3/4 的分析](https://siboehm.com/articles/22/CUDA-MMM)

可以据此设计课堂实验：保持数学操作和 shape 一致，比较 shared load 指令、MIO/short-scoreboard 等相关证据、寄存器占用与时间，再验证寄存器分块是否降低 shared 指令需求。不能仅凭 MIO Throttle 就认定 bank conflict；MIO 队列压力有多种来源。

迁移到 TileLang Tensor Core 路径时，应重新检查实际 MMA、fragment 与 shared 加载方式。SIMT 版本的每线程分块策略不宜直接照搬。

### 7.3 stage 扫描：等待减少，性能仍可能退化

下面是**假设的教学数据，不是本机或外部仓库实测**。shared 数值来自第 5.3 节的简化缓冲估算，occupancy 和延迟仅用于演示推理；真实值必须从指定 GPU 的报告中取得。

| stage | 估算 A/B 缓冲 | 假设 achieved occupancy | 假设独立 benchmark 延迟 |
| ---: | ---: | ---: | ---: |
| 2 | 32 KiB | 50% | 92 μs |
| 3 | 48 KiB | 37.5% | 76 μs |
| 4 | 64 KiB | 25% | 84 μs |

这些数据只能先说明 stage=3 在该 workload 下最快，且更高 occupancy 未对应更低延迟。要解释原因，还需检查实际缓冲分配、occupancy 限制、eligible warp、issue 活跃度和加载/计算相关指标。

如果 2→3 的重叠证据改善，可以支持“增加 stage 隐藏了一部分加载延迟”；如果 3→4 恰好降低每个 SM 的驻留 block 数，才进一步检验资源成本是否导致退化。改变 BM/BN/BK、线程数或 shape 后，需要重新测试。

## 8. 模块二至四：连接 Attention、MoE 与稀疏计算

### 8.1 Batched GEMM → Attention

先用规则 batch/head 实现 `QKᵀ → mask/Softmax → PV`，保证数值语义正确，再连接后续融合 Attention 教程。

| 观察点 | 应采集的证据 | 可能的优化 |
| --- | --- | --- |
| 中间注意力矩阵随序列增长 | 张量大小、分配情况、实际内存流量、完整 Attention 时间 | 分块融合与 online softmax |
| 布局转换成本明显 | 时间线上的 transpose/copy，输入 stride | 改消费布局，减少物化转置 |
| Decode 的 Lq 很小 | grid、KV 读取、各 kernel 时间 | 适配小 query；并行度不足时评估 split-KV 并计入合并成本 |
| GQA 的 K/V 共享 | KV cache 大小、访问与复用方式 | 在 query head 间复用 KV，避免无谓复制 |

标准 FlashAttention 通过减少中间矩阵物化与显存读写实现优化，保留标准 Attention 语义；浮点运算次序可导致数值差异。[FlashAttention 论文](https://arxiv.org/abs/2205.14135)

### 8.2 Grouped GEMM → MoE

固定总 token 数、专家数、Top-K 和专家权重，测试均匀与不均匀路由；比较“逐专家调用 → padding 后批量计算 → grouped”。

记录 `Te` 分布、空专家数、按实际 tile 对齐后的额外工作、kernel 数、调度/重排/合并时间。可用 `max(Te)/mean(Te)` 描述负载偏斜，但它不是性能模型，还要看实际 tile 数和尾部执行情况。

Grouped GEMM 减少小任务独立发射并容纳不同形状；调度方式仍需实测。完整收益必须覆盖 dispatch、专家计算和 combine。MoE 路由稀疏与专家权重内部是否有零，是不同概念。[CUTLASS grouped scheduler](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/grouped_scheduler.md)、[MegaBlocks 论文](https://arxiv.org/abs/2211.15841)

### 8.3 块稀疏矩阵乘 → 稀疏 Attention

先实现块稀疏 `A × W_dense`，在块内复用稠密 GEMM，在块间遍历有效块。固定同一稀疏矩阵，比较 dense reference、朴素稀疏遍历与改进实现。

记录有效块比例、块大小、每行块数分布、索引流量、转换/压缩成本、真实时间，以及实际非零与 dense-equivalent 两种吞吐。静态权重可另报摊销后时间，动态结构则应把构建索引的成本计入目标链路。

Attention 的 QKᵀ 阶段是按选定连接计算输出分数，PV 阶段则是稀疏权重乘稠密 V；不能把两个阶段都理解成“输入换成 CSR”。dense 计算结束后再加 mask，也不会自动省掉前面的乘加。

非结构化稀疏介绍 COO/CSR 的索引成本即可；2:4 用于说明硬件支持的模式、压缩与计算路径。不能把权重置零后继续调用普通 dense GEMM，就宣称获得了稀疏加速。[cuSPARSELt 文档](https://docs.nvidia.com/cuda/cusparselt/index.html)

### 8.4 低精度扩展

在稠密基线建立后，再比较 BF16/FP16 与某一种量化路径。记录权重和元数据字节数、解包/反量化成本、融合情况、误差与组件延迟。存储压缩比例不等于加速比例；小 M 下节省权重流量的收益，也可能被解码成本抵消。[AWQ 论文](https://arxiv.org/abs/2306.00978)

## 9. Profiling 命令与报告保存

### 9.1 先查询本机工具

```bash
nvidia-smi
nvcc --version
nsys --version
ncu --version
ncu --list-sections
ncu --query-metrics --query-metrics-mode all
```

指标不可用时，记录“不支持/未采集”，不能填 0。遇到性能计数器权限错误时，可继续做独立计时与时间线分析，并标明缺少硬件指标。

### 9.2 在已有教学仓库练习

以 [SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA) 为例，先按其 README 配置目标 GPU 的编译架构并完成构建。在该仓库根目录运行：

```bash
make profile KERNEL=3
```

其 Makefile 使用 ncu 采集选定版本。首次熟悉工具后，应按实际 kernel、shape 与调用次数缩小采集范围，避免 benchmark 内的多种大小或预热混入目标报告。

### 9.3 面向后续 TileLang 驱动的命令模板

下面的 `your_benchmark.py` 是占位名，当前目录没有该文件。驱动需要先完成编译、输入准备和预热，明确目标 kernel 名称与调用顺序；命令中的正则也必须替换为实际名称。

```bash
# 定位 kernel、拷贝与 CPU/GPU 间隔。
nsys profile --trace=cuda,nvtx -o gemm_timeline \
  python your_benchmark.py

# 首轮仅采集必要信息；只统计匹配正则的目标 kernel。
# 示例假设该目标 kernel 恰好预热 20 次，随后仍有正式调用。
ncu --kernel-name 'regex:YOUR_GEMM_KERNEL' \
  --launch-skip 20 --launch-count 1 \
  --section SpeedOfLight --section LaunchStats --section Occupancy \
  -o gemm_overview \
  python your_benchmark.py

# 有了具体假设，再增加内存、计算管线与调度分析。
ncu --kernel-name 'regex:YOUR_GEMM_KERNEL' \
  --launch-skip 20 --launch-count 1 \
  --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis \
  --section SchedulerStats --section WarpStateStats \
  -o gemm_detail \
  python your_benchmark.py

# 保留原始报告，再导出便于整理的 CSV。
ncu --import gemm_detail.ncu-rep --page raw --csv > gemm_detail.csv
```

`--launch-skip` 只计数匹配 kernel filter 的调用，不等于跳过整个程序前 20 个 kernel。混合 shape 或自动调优时，优先让驱动只运行一个配置，或使用准确的 NVTX/Profiler API 区间进行过滤。不要把一次 `T.gemm` 误认为必然对应一次独立 kernel launch：它通常是生成 kernel 内部的操作。[NCU CLI 文档](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)

需要源代码关联时，为对应构建启用 line info，并在确认开销可接受后采集 Source Counters/相关源视图。TileLang JIT 的生成代码和缓存也应关联到同一配置，避免用其他版本的源码解释报告。

## 10. 每次实验应该交付什么

每个版本提交一份简短记录，并保存独立 benchmark 数据与相关原始报告。不要预设每个新版本都更快；无收益或退化的结果，同样能解释参数取舍。

```text
实验名称：
代码版本 / Git commit：
GPU / SM 数 / 驱动 / CUDA / TileLang / NCU 版本：
并发负载 / 频率与功耗条件：
M,N,K / dtype / accum dtype / output dtype / layout / stride：
BM,BN,BK / threads / stages：
输入与权重复用方式 / CUDA Graph 模式：
计时区间 / 预热 / 调用次数 / 样本数：
NCU 目标 kernel / replay 与 cache 设置：

问题：为什么选择这个 case，原版本慢在哪里？
证据：哪些指标和代码位置支持该判断？
假设：只修改什么，希望改善什么？
结果：延迟中位数与波动、资源占用、目标指标、数值误差。
解释：指标是否支持原假设？有没有其他可能原因？
决定：保留、限制适用 shape，还是回退？
附件：benchmark 原始数据、nsys/ncu 报告、生成代码（按需）。
```

第一轮验收包括：Linear 的大 M/小 M 对照、至少一次有指标支持的改进、一次解释清楚的退化，以及在完整组件中复测收益。例如，当 GEMM 只占组件时间的 30% 时，即使它加速 2 倍，按固定其余成本的估算，组件也只加速约 `1/(0.7+0.3/2)=1.18` 倍。完成单 kernel 优化后，需要回到模型组件判断价值。
