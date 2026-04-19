# QSHNet Risk-Aware Quaternion Gate Ceiling Design

## 背景

在当前 `variable residual only` 版本上，`USHCN itr=5` 可接受，但 `itr=10` 仍出现明显厚尾坏轮：

- 好轮可达 `0.1635 ~ 0.1680`
- 坏轮可达 `0.2021 / 0.2204 / 0.2374`

对 `itr=10` 的好轮与坏轮 checkpoint 对比后，坏轮的共同特征不是 event 分支更强，而是：

- `retain_log_scale` 更高
- `membrane_proj.weight norm` 更高
- `quat_gate.weight norm` 更高
- `quat_h2n` 的 `i/j/k` 范数更高

这说明坏轮更像是 `retain + quaternion` 的过增强收敛，而不是 event 分支单独失控。

## 目标

只改一个核心因素：限制高风险状态下的 quaternion gate 有效增益。

目标优先级：

1. 将 `USHCN itr=10` 的最坏轮压到 `<= 0.18`
2. 可以接受均值轻微上升
3. 不改 event 主线、不改 retain 主线、不改训练超参数

## 方案

保留现有 quaternion 路径：

- 先计算原始 `alpha_raw = sigmoid(quat_gate(...))`
- 再引入风险相关的缩放 `cap_scale`
- 最终使用 `alpha = alpha_raw * cap_scale`

其中风险信号直接复用现有 fused-risk 诊断：

- `risk = 1 - adaptive_ratio_max_mean / coupled_residual_ratio_max`

含义：

- `risk` 越高，说明当前 fused cap 越紧，当前状态越危险
- 高风险状态下，应该降低 quaternion 残差的入口增益

建议的第一版参数：

- `quat_risk_ceiling_min = 0.65`
- `cap_scale = 1 - (1 - quat_risk_ceiling_min) * risk`

这样可以保证：

- 低风险时：`cap_scale ≈ 1.0`
- 高风险时：`cap_scale` 最低约为 `0.65`
- 属于温和压制，而不是硬截断

## 修改范围

只涉及：

- `models/QSHNet.py`
- `tests/models/test_QSHNet.py`

不修改：

- event 注入逻辑
- fused residual 路径
- retain 路径
- 训练配置和超参数

## 预期效果

理想结果：

- `itr=10` 的坏轮上界明显回落
- `0.22+ / 0.23+` 的厚尾坏轮消失
- 好轮轻微变差可以接受，只要整体尾部被压住

可能代价：

- 最优单轮不再达到当前最低值
- 均值可能轻微上升

## 验证

本地只验证：

1. 单元测试
2. `USHCN itr=10`

判定标准：

1. 第一优先：`max_mse <= 0.18`
2. 第二优先：均值不要显著劣化
3. 若坏轮上界下降但均值轻微升高，仍视为方向正确
