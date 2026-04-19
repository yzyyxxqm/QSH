# QSH-Net 耦合残差上界设计

> 日期：2026-04-18
> 目标：在不破坏 `HumanActivity / P12 / MIMIC_III` 已确认可用性的前提下，优先压制 `USHCN` 上由 `retain/quaternion` 与 `event` 耦合放大导致的坏轮。

## 1. 背景

当前统一主线仍是：

- `eventscalecap_main`
- `eventscalecap_itr10`

当前工作区保留候选是：

- `eventdensvar_main`

服务器四数据集正式验证已经说明：

- `HumanActivity`：通过
- `P12`：通过
- `MIMIC_III`：基本可接受，但仍有单轮失稳
- `USHCN`：未通过

因此下一轮结构优化不再追求继续改善 `HumanActivity`，而是只针对 `USHCN` 的坏轮压制做单因素试验。

## 2. 当前问题的结构判断

当前更合理的问题归因不是：

- `event` 总量单独过大
- 或 `variable density` 单独过大

而是：

- `event` 注入进入活跃主状态后，与 `retain` 调制和 quaternion 残差共同形成放大链条
- 在 `USHCN` 这种高方差数据上，该耦合残差会在部分 seed 中被异常放大

因此下一轮不再直接继续压：

- `event_scale`
- `event_delta`
- `variable density penalty`

而是转向约束：

- `event` 注入与活跃主状态形成后的相对耦合残差

## 3. 方案 A：Coupled Residual Bound

### 3.1 核心思想

只限制“耦合后残差”的相对幅度，而不是在耦合前直接压 `event` 本身。

更具体地说：

1. 保留当前 `eventscalecap_main / eventdensvar_main` 的 `event` 注入机制。
2. 保留当前 quaternion residual 的表达方式。
3. 在 `event` 注入进入主状态之后、并继续参与后续活跃分支更新之前，对“相对主状态的额外残差”增加一个软 ratio cap。

目标不是抹掉 `event`，而是防止：

- `event` 注入
- `retain` 保留主路径
- quaternion refinement

在高方差样本上联合作为一个放大器工作。

### 3.2 与旧 `eventrescap_main` 的本质区别

这次方案必须明确区别于已经失败的 `eventrescap_main`。

旧 `eventrescap_main` 的问题是：

- 直接对 `event residual` 全局做相对主干范数硬约束
- 作用点过早
- 本质上仍然是在压 `event` 自身
- 结果是 `USHCN` 上有效注入和异常放大一起被压掉

本方案的区别是：

1. 不改变 `event_scale` 的现有语义。
2. 不在 `event_delta` 形成时动手。
3. 不做全局 `event residual` 比例上界。
4. 只在 `event` 已经注入主状态之后，约束“耦合后额外残差”相对主状态的比例。

也就是说，这一版压的是：

- `coupled residual`

而不是：

- `raw event residual`

## 4. 作用点设计

### 4.1 结构插入位置

当前 `QSHNet.py` 的关键路径是：

1. `SpikeRouter` 产出：
   - `obs_base`
   - `obs_event`
2. `event_delta` 经 `temporal / variable` 聚合与归一化后，
   通过 `apply_event_injection(...)` 注入 `temporal_hyperedges_updated / variable_hyperedges_updated`
3. 再经过 `hyperedge -> node`
4. 再做 quaternion residual refinement

本方案的推荐插入点不是 `SpikeRouter` 内部，也不是 `compute_event_scale()` 内部，而是：

- `apply_event_injection(...)` 之后可直接观测到的“注入后主状态”
- 或 `h2n_out` 形成前后、可显式分解“主状态 + 耦合残差”的位置

推荐优先实现为：

- 先在 `apply_event_injection(...)` 内部完成

原因：

1. 这里能最清楚地区分：
   - `main_state`
   - `event_delta`
   - `injected_state`
2. 不会把 quaternion 支路本身直接改坏。
3. 改动范围最小，便于做单因素验证。

### 4.2 残差定义

定义：

- `main_state`：当前 layer 的原始 hyperedge 主状态
- `injected_state = main_state + event_scale * event_delta`
- `coupled_residual = injected_state - main_state`

当前实现中，`coupled_residual` 在数值上等于 `event_scale * event_delta`，但设计语义上必须按“注入后残差”来处理，而不是再把它当成单独的 `event` 原件。

下一轮的约束形式是：

- 只限制 `||coupled_residual|| / ||main_state||`

### 4.3 上界形式

采用软 ratio cap，而不是硬截断：

1. 先计算：
   - `main_norm = ||main_state||`
   - `residual_norm = ||coupled_residual||`
2. 给定：
   - `coupled_residual_ratio_max`
3. 当 `residual_norm <= ratio_max * main_norm` 时，不做处理。
4. 当超过时，只按比例缩小 `coupled_residual`，而不是直接裁成常数或直接丢弃。

输出：

- `bounded_state = main_state + bounded_coupled_residual`

这样保持：

- 方向不变
- 只压幅值
- 恒等初始化不受影响

## 5. 最小改动原则

本轮必须严格遵守“每次只改一个核心因素”。

因此不改下面这些部分：

1. `event_scale_max = 0.12`
2. `event_density_baseline`
3. `temporal_event_density_penalty_max`
4. `variable_event_density_penalty_max`
5. `temporal_event_norm / variable_event_norm`
6. `quat_residual_ratio_max`
7. `retain_strength_max`

本轮唯一新增因素应是：

- `coupled_residual_ratio_max`

并且它只作用于：

- `apply_event_injection(...)`

## 6. 推荐初始参数

为了避免首轮就把有效注入一起压坏，第一版建议用偏宽松的初值：

- `coupled_residual_ratio_max = 0.20`

如首轮观察到：

- `USHCN` 坏轮仍然明显
- 但 `HumanActivity` 无损

再考虑第二轮微调为：

- `0.16`
- 或 `0.12`

但这必须作为后续独立试验，不能在首轮同时扫描。

## 7. 诊断指标

本轮除了最终 MSE，还必须记录新的结构诊断量。

建议新增：

1. `coupled_residual_norm_mean`
2. `coupled_residual_ratio_mean`
3. `coupled_residual_ratio_max`
4. `coupled_residual_clip_rate`

其中：

- `clip_rate` 表示当前 batch 中有多少位置触发了 ratio cap

希望看到的现象是：

1. `HumanActivity`
   - `clip_rate` 很低或接近 0
2. `USHCN`
   - `clip_rate` 在坏轮风险较高的 batch 上适度升高
   - 同时最终 `itr=10` 方差下降

如果出现：

- `HumanActivity` 上 `clip_rate` 也很高

则说明该约束过早伤到了简单数据。

## 8. 验证顺序

### 8.1 本地快速筛查

先跑：

1. `HumanActivity itr=3`
2. `USHCN itr=5`

筛查标准：

1. `HumanActivity` 不明显差于当前 `eventdensvar_main`
2. `USHCN` 不能比当前 `eventdensvar_main itr=5` 更差
3. 诊断日志能看到该 ratio cap 确实只在局部起作用，而不是全局普遍压制

### 8.2 正式判断

通过本地筛查后，再跑：

- `USHCN itr=10`

本轮成功门槛已经固定为：

- `mean <= 0.178`
- `std <= 0.02`

### 8.3 跨数据集确认

如果 `USHCN itr=10` 达标，再送：

1. `P12 itr=5`
2. `MIMIC_III itr=5`

要求：

- 不明显退化

## 9. 风险与失败判据

### 9.1 主要风险

1. 上界过紧，导致退化成保守版本。
2. 约束位置虽然在耦合后，但数值效果仍等价于“继续压 event”。
3. `USHCN` 的坏轮真正来源并不在 hyperedge 注入后，而在更后面的 quaternion refinement。

### 9.2 明确失败判据

满足任意一条即可判定本方案失败：

1. `HumanActivity itr=3` 明显退化。
2. `USHCN itr=5` 已经明显弱于当前 `eventdensvar_main`。
3. `USHCN itr=10` 无法同时满足：
   - `mean <= 0.178`
   - `std <= 0.02`
4. `clip_rate` 在所有数据集、所有 batch 上都很高，说明它在做粗暴全局压制。

## 10. 推荐结论

下一轮最值得先验证的唯一假设是：

> `USHCN` 的坏轮主要来自 `event` 注入进入活跃主状态后的耦合放大；如果只对耦合后残差施加相对主状态的软上界，而不直接压 `event` 本身，则有机会在保住 `HumanActivity / P12 / MIMIC_III` 的同时，把 `USHCN itr=10` 拉回 `mean <= 0.178` 且 `std <= 0.02`。 
