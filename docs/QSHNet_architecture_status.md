# QSH-Net 当前模型架构状态总览

> **最后更新：** 2026-04-21
> **用途：** 作为后续继续做结构试验时的单一架构状态基准。优先级高于零散实验记录与口头结论。

## 1. 当前真实动力学

当前 QSH-Net 的真实训练动力学，不应再表述为：

- `Spike + Event + Quaternion` 已经稳定协同

更准确的表述是：

- `HyperIMTS 主干 + retain 调制 + quaternion 残差 + 已被唤醒但仍不稳定的 event 注入`

其中：

- 超图主干仍然承担主要结构建模；
- `retain` 与 quaternion 仍然是持续活跃的主增强路径；
- `event` 已经不再是旧 M1 中的死分支；
- 但 `event` 仍然不是稳定可靠的主收益来源；
- `USHCN` 的坏轮更像是真实活跃分支在高方差数据上被放大，而不是 `event` 单独失效。

## 2. 已确认的基础事实

### 2.1 旧 M1 的真实问题

旧 M1 已通过本地参数诊断确认：

- `event_proj.weight norm = 0.0`
- `event_residual_scale = 0.0`
- 在 `USHCN` 与 `HumanActivity` 上都成立

这说明：

- 旧 M1 的 `event residual` 分支最初是死分支；
- 当时真正持续学习的是：
  - `retain_log_scale`
  - `membrane_proj.weight`
  - `quat_gate`
  - `quat_h2n`

### 2.2 当前代码中的真实活跃路径

当前持续活跃、可以在诊断里稳定观测到的部分仍主要是：

- `retain`
- quaternion residual
- 已被唤醒的 `event` 注入

但当前更应把 `event` 视为：

- 已经参与训练；
- 但收益不稳定；
- 且容易在 `USHCN` 这类高方差数据上与其他活跃分支一起放大尾部风险。

## 3. 当前代码主线结构

当前代码工作区的默认主线，应视为：

- `qsh_rescorrnorm100_cap003` 这一类 residual-correction 变体

其核心结构包括：

1. **HyperIMTS 主干保留**
- observation node / temporal hyperedge / variable hyperedge 三层表示未被推翻；
- 当前所有有效试验都建立在“不破坏 HyperIMTS 主干”的前提上。

2. **SpikeRouter 仍为连续路由而非硬脉冲**
- `retain_gate` 决定主路径保留强度；
- `event_gate` 决定事件路径注入强度；
- `membrane_proj` 仍然是 route 动力学的关键参数来源之一。

3. **event 支路已被唤醒且受控**
- `event_residual_scale` 使用 `sigmoid` 语义；
- `event_scale` 带有温和上界；
- temporal / variable event 注入有独立归一化；
- 当前代码里仍保留 `event_proj_norm_cap` 这种实验开关，但默认不启用，也不属于当前保留主线。

4. **quaternion 残差保留**
- `quat_h2n` 与 `quat_gate` 仍参与真实学习；
- quaternion residual 仍是当前有效增强项之一；
- 但最近几轮实验已经说明，单独去压 quaternion 输出比例不是根因级修复。

5. **输出端 residual correction 头已存在**
- 当前支持：
  - residual context normalization；
  - correction cap；
  - confidence gate；
  - self gate；
- 但最近一轮实验已经说明，输出端门控不是 `USHCN` 坏轮的根因级修复方向。

## 4. 当前最重要的结构判断

截至 2026-04-21，最准确的架构判断应为：

1. 当前模型仍然不是稳定成熟的「三元素完全协同」结构。
2. `event` 分支已经比旧 M1 更活跃，但仍不是稳定可靠的增益主力。
3. `retain + quaternion + HyperIMTS backbone` 仍构成主导动力学。
4. 输出端再给 residual correction 叠加门控，不能解决 `USHCN` 坏轮。
5. 统一收紧 `innovation_budget` 这类共享预算，也不能在保住好轮的同时压掉坏轮。

## 5. 最近一轮结构试验的收敛结论

本轮围绕 `qsh_rescorrnorm100_cap003` 主线，已经完成以下单因素试验：

### 5.1 `event_proj_norm_cap`

已尝试：

- `epcap5`
- `epcap6`

结论：

- 简单裁剪 `event_proj` 的有效范数并不能解决 `USHCN` 的坏轮；
- 它会同时伤到原本表现良好的 seed；
- 说明问题不在 `event_proj` 权重范数单点失控。

### 5.2 `qsh_residual_correction_self_gate`

结果：

- `HumanActivity`: `0.041475 ± 0.000215`
- `USHCN`: `0.190852 ± 0.037656`

结论：

- 对 `HumanActivity` 只有极小影响；
- 对 `USHCN` 没有压掉坏轮，反而把两个 seed 拉到 `0.23+`；
- 说明自门控 residual correction 不是根因级修复。

### 5.3 `qsh_residual_correction_confidence_gate`

结果：

- `USHCN`: `0.207944 ± 0.039097`

结论：

- 只允许高置信 residual correction 放大，并没有改善坏轮；
- 反而整体退化；
- 说明问题不在 residual correction 的「置信约束」缺失。

### 5.4 `innovation_budget_init = 0.75`

结果：

- `USHCN`: `0.194180 ± 0.036793`

结论：

- 统一收紧 event 注入与 quaternion residual 的共享预算，会误伤好轮；
- 但照样挡不住坏轮；
- 说明「整体收紧活跃分支」不是正确方向。

## 6. 当前保留与否决边界

### 6.1 当前仍可保留的主线认知

- `qsh_rescorrnorm100_cap003` 仍是当前直接工作的局部主线；
- 它有 4 个非常强的好轮；
- 但仍存在少数坏轮；
- 当前还没有找到一个单因素改动，既能保住这 4 个好轮，又能稳定压掉坏轮。

### 6.2 当前已明确否决的方向

最近这轮之后，不应继续优先尝试：

- 再扫 `event_proj_norm_cap` 阈值；
- 在输出 residual correction 头上继续叠门控；
- 统一收紧 `innovation_budget`；
- 把问题简单归结为输出头没调好。

## 7. 后续若继续推进，应遵守的边界

1. 先假设问题在活跃主路径的内部比例失衡，而不是输出头没有加够 gate。
2. 优先考虑更接近 route / propagation / quaternion 有效残差占比的根因级约束。
3. 每次只改一个核心因素。
4. `USHCN` 不再以“最坏轮上界尽量低”为唯一目标，而是至少看 `itr>=5` 中大部分 seed 是否优于 `HyperIMTS` 论文/项目参考。
5. 后续主优化目标应转向非 `USHCN` 数据集的稳定收益，而不是为了少量 `USHCN` 坏轮做过度保守化。
