# QSH-Net 模型演进记录

> **最后更新：** 2026-04-21
> **说明：** 这份文档保留阶段性演进脉络，但当前应优先以最新结构判断为准，不再把旧阶段结果直接视为当前主线。

## 1. 早期阶段的核心教训

QSH-Net 的早期多轮尝试已经反复证明：

1. 不能破坏 HyperIMTS 主干；
2. 新增强分支如果不能从安全语义出发，会直接把基线拖坏；
3. 旧 M1 中 `event` 曾经是死分支；
4. 真正长期活跃的，一直是 `retain + quaternion + HyperIMTS backbone`。

这些结论今天仍然成立。

## 2. 中期阶段：从「event 死分支」到「event 被唤醒」

之后的一轮结构试验链，核心贡献不是单纯把数值拉高，而是确认了：

- `event` 已经不再是死分支；
- `event` 可以被稳定地接入主干；
- 但 `event` 仍然不是稳定可靠的主收益来源；
- `USHCN` 的坏轮不是单纯的 `event` 学不起来，而是活跃主路径在高方差数据上被放大。

## 3. 近阶段：从大版本候选退回到根因级诊断

最近一阶段，工作重点已经从“继续造新主线版本”切换为：

- 以 `qsh_rescorrnorm100_cap003` 这一类 residual-correction 版本为直接工作起点；
- 针对 `USHCN` 的坏轮做根因级诊断；
- 每次只改一个核心因素；
- 先看本地 `USHCN` 是否通过，再决定是否扩展到其他数据集或服务器。

## 4. 2026-04-21 之前最近完成的试验链

### 4.1 `event_proj_norm_cap`

目的：

- 验证坏 seed 是否主要由 `event_proj` 权重范数漂高导致。

结果：

- `epcap5` 失败；
- `epcap6` 更差；
- 说明简单裁剪 `event_proj` 有效范数不是正确方向。

### 4.2 输出端 residual correction 门控

已测：

- `selfgate`
- `confgate`

结果：

- `selfgate` 在 `HumanActivity` 基本持平，但在 `USHCN` 上失败；
- `confgate` 在 `USHCN` 上明显更差。

结论：

- 坏轮不是因为输出 residual correction 没有被 gate 好；
- 输出端再叠门控不是根因级修复。

### 4.3 统一收紧共享创新预算

已测：

- `innovation_budget_init = 0.75`

结果：

- `USHCN = 0.194180 ± 0.036793`

结论：

- 统一收紧 event 与 quaternion 的共享预算，会误伤好轮；
- 但依然挡不住坏轮；
- 说明问题不在整体预算过宽。

### 4.4 遗漏补记：`routebudget / outputfusion / spikeselectprop_adapt`

在最近整理工作区结果时，补回了几组此前没有完整写进主文档的本地实验：

1. `routebudget`
- `qsh_routebudget_floor025_ushcn_local = 0.169866 ± 0.014330`
- `qsh_routebudget_retainbudget_ushcn_local = 0.190289 ± 0.029010`
- 说明 `floor025` 明显优于 `retainbudget`，这条链有过可接受的 `USHCN` 局部结果。

2. `outputfusion / coredecoderinit`
- `qsh_outputfusion005_coredecoderinit_human_local = 0.041545 ± 0.000235`
- `qsh_outputfusion005_coredecoderinit_ushcn_local = 0.223903 ± 0.049122`
- `qsh_coredecoder_rescorr_main_local` 在 `HumanActivity` 上为 `0.041477 ± 0.000186`，但在 `USHCN` 上退化到 `0.253231 ± 0.023152`
- 说明「保住 HyperIMTS 起点，再叠输出头」当前还不是安全主线。

3. `spikeselectprop_adapt`
- `spikeselectprop_adapt025_qbias4_local`：`HumanActivity = 0.041735 ± 0.000144`，`USHCN = 0.164609 ± 0.002637`
- `spikeselectprop_adapt025_itr10`：`HumanActivity = 0.041747 ± 0.000165`，`USHCN = 0.165544 ± 0.005667`
- `spikeselectprop_quat015_local = 0.175164 ± 0.019627`
- 说明传播选择自适应链曾给出比最近 residual-correction 近邻更强的本地 `USHCN` 结果。

## 5. 当前收敛判断

截至 2026-04-21，最稳妥的演进结论是：

1. 当前没有新的单因素改动能够替代 `qsh_rescorrnorm100_cap003` 起点。
2. 当前问题不在输出头，也不在简单整体收缩。
3. 当前更像是活跃主路径内部比例失衡。
4. `spikeselectprop_adapt025` 与 `routebudget_floor025` 应保留为已做过、且值得随时回看的历史本地候选。
5. `coredecoderinit / outputfusion` 这条链当前不应再被表述为“默认下一条主线”。
6. 如果继续推进，应更靠近：
- route / propagation 机制；
- quaternion 有效残差占比；
- 活跃主路径之间的内部比例约束。

## 6. 当前不应再沿用的旧叙事

不应再直接沿用下面这些旧阶段叙事：

- `eventgateconst` 是当前默认主线；
- `spikeselectprop_res005` 仍是当前工作区主线；
- 最近的主要工作还是围绕 `eventgateconst / spikeselectprop` 展开；
- `qsh_coredecoder_rescorr_main` 仍是当前默认下一条服务器主线；
- 继续在服务器上扩展最近这批本地失败近邻有意义。

这些表述都已经落后于当前真实状态。
