# QSH-Net 模型演进记录

## 基线参照

HyperIMTS 论文报告的 MSE（5轮均值±标准差）：

| 数据集 | MSE |
|--------|-----|
| P12 | 0.2996 ± 0.0003 |
| HumanActivity | 0.0421 ± 0.0021 |
| USHCN | 0.1738 ± 0.0078 |

---

## v1~v3：初始实现阶段

### v1：完整实现设计文档全部 5 个模块
- 模块1：四元数线性层（Kronecker 块矩阵 Hamilton 积 + 分离激活）
- 模块2：脉冲状态空间模型（ZOH 离散化 + 自适应替代梯度 + Gray Code RPE）
- 模块3：有向因果超图（无填充节点表征 + HGNN-AS 软剪枝 + 因果拉普拉斯）
- 模块4：联合时频建模（自适应分块 + NUDFT + iNUDFT 季节性偏差）
- 模块5：DBLoss（EMA 趋势-季节分解损失）

**结果：** GPU OOM（3.6GB 显存不足），batch_size 从 32 降到 2 仍然 OOM。Loss 爆炸到 1e7 级别。

### v2~v3：速度优化
- S-SSM：从逐步 for loop 改为 log-space cumsum 并行化
- 因果拉普拉斯：从 O(N²) 密集矩阵改为 O(N) cumsum
- 去掉 node_self_attention

**结果：** 能跑了，但 MSE 远高于 HyperIMTS。

---

## v4~v5：轻量化重设计

### v4：极简架构
**结果（1轮）：** P12=0.316, Human=0.075, USHCN=0.184

### v5：增强表达力
**结果：** MSE 几乎没有下降。

**教训：** 在弱基础上加组件无法弥补核心消息传递机制的差距。

---

## v6~v7：恢复 HyperIMTS 核心

### v6：恢复 HyperIMTS 级别的消息传递
**结果（1轮）：** P12=0.303, Human=0.044, USHCN=0.228

### v7：完整复刻 HyperIMTS + 外挂组件
**结果（1轮）：** P12=0.303, Human=0.043, USHCN=0.228

**教训：** 外挂的 QuaternionBlock 和 SpikingGate 对 MSE 没有贡献。

---

## v8：深度融合尝试

**结果（1轮）：** 几乎没有变化，Human 略微退步。

**教训：** 四元数替换 Linear 引入了过强的结构约束；因果掩码改变了超图拓扑引入噪声。

---

## 消融实验阶段

### 系统性消融（5 个配置 × 3 个数据集）

| 配置 | P12 MSE | Human MSE | USHCN MSE |
|------|---------|-----------|-----------|
| A: 全部开启 | 0.3004 | 0.0464 | 0.1814 |
| B: w/o 四元数 | 0.3008 | 0.0436 | 0.1649 |
| C: w/o 脉冲 | 0.3009 | 0.0440 | 0.2244 |
| D: w/o 因果掩码 | 0.2992 | 0.0493 | 0.2074 |
| **E: 纯 HyperIMTS** | **0.2994** | **0.0413** | **0.1671** |

**关键发现：** 纯 HyperIMTS 复刻（E）在所有数据集上都是最好或接近最好的。

---

## 有机重设计阶段（2026-04-12）

### 第一步：Context-Aware Spike Selection 重设计
利用超图的 `variable_incidence_matrix` 计算每个变量的均值上下文，标量膜电位 `Linear(2D, 1)` 替代旧版 `Linear(D, D)`。

### 第二步：语义四元数融合（失败）
**结果：** USHCN MSE 从 0.168 恶化到 0.221+。

### 第三步：四元数残差精炼（改善但不稳定）
**结果：** 单次最优 0.165，但 5 轮均值 0.211 ± 0.040。

### 第四步：恒等初始化（突破！）
**关键发现：** 所有 QSH 增强组件必须从精确恒等开始。
**结果（itr=3，无 weight decay）：** 均值 0.181 ± 0.009

### 第五步：AdamW + Weight Decay（稳定化）
**结果（itr=5，全数据集评估 2026-04-14）：**

| 数据集 | 论文 HyperIMTS | **QSH-Net** | 改善 |
|--------|---------------|-------------|------|
| **MIMIC_III** | 0.4259 ± 0.0021 | **0.3933 ± 0.0060** | **-7.7%** ✅ |
| **MIMIC_IV** | 0.2174 ± 0.0009 | **0.2157 ± 0.0022** | **-0.8%** ✅ |
| **HumanActivity** | 0.0421 ± 0.0021 | **0.0416 ± 0.0003** | **-1.2%** ✅ |
| P12 | 0.2996 ± 0.0003 | 0.3006 ± 0.0013 | +0.3% ⚠ |
| USHCN | 0.1738 ± 0.0078 | 0.1870 ± 0.0277 | +7.6% ❌ |

各轮详情：
- MIMIC_III: 0.388, 0.389, 0.402, 0.400, 0.388
- MIMIC_IV: 0.2155, 0.2140, 0.2190, 0.2145
- HumanActivity: 0.0413, 0.0419, 0.0413, 0.0415, 0.0417
- P12: 0.3009, 0.2998, 0.3027, 0.3001, 0.2996
- USHCN: 0.1696, 0.1687, 0.2361(异常), 0.1617, 0.1991

---

## 核心教训总结

### 1. 恒等初始化是最重要的单一改进
新组件从随机初始化开始 = 向基线模型注入噪声。唯一安全的做法是从精确恒等开始。

### 2. 加法精炼 > 插值融合
`linear + α*quat` 不会减弱原始信号。

### 3. Weight decay 对高方差数据集至关重要
AdamW (wd=1e-4) 将 USHCN 5-run std 从 0.040 降到 0.011。

### 4. 不要改变超图核心
v1~v8 的反复失败证明：任何改变 HyperIMTS 核心消息传递机制的尝试都会伤害性能。

---

## USHCN 基础模型优化实验（2026-04-13，已全部回退）

### 全面对比

| 版本 | 均值 | std | 中位数 | 好运行率 |
|------|------|-----|--------|----------|
| HyperIMTS 同环境 | 0.2016 | 0.0352 | 0.1745 | 3/5 |
| **原始 QSH-Net** | **0.1862** | 0.0268 | **0.1725** | **4/5** |
| A: LayerNorm | 0.2015 | 0.0234 | 0.1997 | 3/5 |
| A+B: LN+WarmCos | 0.2843 | 0.0710 | 0.2522 | 0/5 |
| A+C.1: LN+Time | 0.2340 | 0.0277 | 0.2417 | 1/5 |
| A+C.2: LN+Drop | 0.2283 | **0.0170** | 0.2385 | 1/5 |

### 核心结论

1. **正则化越强 → 越稳定但越差**
2. **双峰 ≠ bug，而是 feature**
3. **USHCN 上不要动基础模型**
4. **Early stopping 与慢衰减 LR 不兼容**

所有改动已回退，恢复到产出四数据集结果的原始版本。

---

## 待解决问题

1. USHCN 训练不稳定：itr=5 中出现 0.236 异常轮
2. P12 略高于基线 0.3%
3. 四元数门控 alpha ≈ 0.047 几乎不动——可能需要更高的初始值或独立学习率
4. 脉冲 fire_rate 跨 run 变化很大
5. **MIMIC_III 改善 7.7% 的机理需要深入分析**

---

## 有机融合重设计实验（2026-04-14~15，已全部回退）

### 背景与动机

旧版 QSH-Net 的 Spike 和 Quaternion 是"贴"在超图上的，不是"长"在超图里的：
- SpikeSelection 只在消息传递前做标量加权（fire_rate=16%, attenuation=98.2%，区分度仅 1.8%）
- QuaternionLinear 只在融合后做残差补充（alpha≈0.047 几乎不动）
- 超图的消息传递本身完全没变

目标：让四元数和脉冲有机融合进超图消息传递的核心机制中。

### 设计 A：QuaternionStructuredFusion + SpikeTemperature

**QuaternionStructuredFusion**（替换 Linear(3D, D)）：
- obs/tg/vg 分别投影为四元数的 i/j/k 分量，线性融合作为实部 r
- Hamilton 积（Kronecker 块矩阵）自动产生跨源交互：时间×变量、变量×节点、节点×时间
- 输出取实部

**SpikeTemperature**（替换 SpikeSelection）：
- 脉冲信号调制 n2h 注意力的温度（而非标量加权）
- 偏离变量上下文 → 低温度 → 尖锐注意力；符合上下文 → 高温度 → 平滑注意力

### 实验 A1：D/4 维分量 + D×D Kronecker（首版）

每个四元数分量投影到 D/4 维，Kronecker 块矩阵 (D×D)，子矩阵 (D/4×D/4)。

**USHCN itr=5 结果：** 0.235, 0.246, 0.225, 0.275, 0.169 → 均值 0.230 ± 0.037

**失败原因：** D/4 瓶颈——每个分量只有 64 维（D=256），三源融合信息从 256 维压缩到 64 维，丢失 75% 信息。与之前"语义四元数"失败的原因完全一致。

### 实验 A2：全 D 维分量 + 4D×4D Kronecker

每个分量保持完整 D=256 维，Kronecker 块矩阵 (4D×4D)=(1024×1024)，子矩阵 (D×D)=(256×256)。输出取实部（前 D 维）。

**USHCN itr=3 结果：** 0.166, 0.249, 0.182 → 均值 0.199 ± 0.037

**问题：** 参数量爆炸（4×D² = 262K per layer），且 `proj_r` 的随机初始化破坏了恒等初始化——训练起点不是纯 HyperIMTS。

### 实验 A3：修复恒等初始化（Linear 主路径 + Hamilton 积残差 + gate=0）

保留原始 `Linear(3D, D)` 作为主路径，Hamilton 积作为残差补充，gate 初始化为 0（tanh(0)=0）确保精确恒等初始化。

**USHCN itr=3 结果（含 SpikeTemperature）：** 0.193, 0.212, 0.195 → 均值 0.200 ± 0.010

**分析：** 方差大幅降低（0.010 vs 0.037），恒等初始化修复有效。但均值 0.200 仍比旧版（0.187）差。

### 实验 A4：消融——去掉 SpikeTemperature

只保留 QuaternionStructuredFusion，n2h 注意力回到标准 MultiHeadAttentionBlock。

**USHCN itr=3 结果：** 0.215, 0.161, 0.164 → 均值 0.180 ± 0.030

**关键发现：SpikeTemperature 在伤害性能**（去掉后从 0.200 改善到 0.180）。温度调制在 USHCN（仅 5 变量）上区分度太低，反而引入噪声。

### 实验 A4 全数据集评估

| 数据集 | QuatFusion only | 旧版 QSH-Net | HyperIMTS 论文 |
|--------|----------------|-------------|---------------|
| HumanActivity (itr=5) | 0.0417 ± 0.0003 | 0.0416 ± 0.0003 | 0.0421 ± 0.0021 |
| P12 (itr=5) | 0.3021 ± 0.0015 | 0.3006 ± 0.0013 | 0.2996 ± 0.0003 |
| MIMIC_III (itr=5) | 0.4015 ± 0.0115 | **0.3933 ± 0.0060** | 0.4259 ± 0.0021 |
| USHCN (itr=3) | 0.180 ± 0.030 | 0.187 ± 0.028 | 0.174 ± 0.008 |

**结论：** QuaternionStructuredFusion 在 MIMIC_III 上退步（0.401 vs 0.393），方差翻倍（0.012 vs 0.006）。其他数据集基本持平。

### 核心教训

1. **D/4 瓶颈是致命的**：四元数的 4 分量结构天然要求将 D 维切成 4 份，这在 D=256 时丢失太多信息
2. **恒等初始化必须精确**：任何新的 Linear 层（如 proj_r）如果不和原始 HyperIMTS 的 Linear 共享权重，就会破坏恒等初始化
3. **SpikeTemperature 在低变量数数据集上有害**：温度调制需要足够的变量多样性才能产生有意义的区分
4. **Hamilton 积的结构约束不一定优于自由 Linear**：在 MIMIC_III 上，旧版的 `Linear(3D,D) + α*QuatLinear(linear_out)` 比新版的结构化融合效果更好
5. **旧版的"拼接"设计虽然不够优雅，但恒等初始化 + 加法残差的安全性是其成功的关键**

### 决策

所有新设计实验已回退，恢复到旧版 QSH-Net（SpikeSelection + QuaternionLinear 加法精炼）。该版本在 5 个数据集中 3 个超越 HyperIMTS 基线（MIMIC_III -7.7%, HumanActivity -1.2%, MIMIC_IV -0.8%），是目前效果最好的版本。

---

## 分层协同型 M1 之后的结构试验链（2026-04-16，当前主线）

### 当前背景

旧 M1 的实现方向没有回退，但针对其真实训练动力学，后续已经完成一轮更细的结构试验链。

此时最关键的事实不是「某个超参数没调好」，而是：

1. 旧 M1 的真实训练动力学并不是「Spike + Event + Quaternion 协同」。
2. 旧 M1 更接近 `HyperIMTS + retain 调制 + quaternion 残差`。
3. 如果希望保住论文里的「三元素统一框架」，就必须先让 `event` 分支从死分支变成真实可训练、真实参与训练的分支。

### 阶段 1：bounded-gate 保守稳定化（已回退）

---

## USHCN 尾部压制试验补充记录（2026-04-19）

这一轮工作的目标不是继续做常规超参数扫描，而是回答一个更具体的问题：

- 在当前 `variable residual only + adaptive fused-cap` 主线已经无法稳定压住 `USHCN itr=10` 坏轮的前提下，下一步到底该继续碰哪里？

围绕这个问题，已经完成以下 4 个单因素结构试验。

### 1. `eventgateconst`：解除 `event_gate` 对 `route_logit` 的直接耦合

改动：

- 将
  - `event_gate = exp(event_log_scale) * sigmoid(route_logit)`
- 改为
  - `event_gate = 0.5 * exp(event_log_scale)`

语义：

- route 仍然控制 `retain`
- route 仍然参与 density 统计
- 但 route 不再直接放大 event 注入幅度

**USHCN itr=10：**

- `0.1561, 0.1905, 0.2156, 0.1660, 0.1670, 0.1788, 0.1702, 0.1697, 0.1708, 0.1739`
- 均值 `0.17587`
- std `0.01571`
- max `0.21558`

**结论：**

- 这是最近一轮尾部压制试验里唯一明确有效的方向
- 它没有解决全部坏轮
- 但它显著优于 `routebound075 / eventprojgrad2 / routeconfvar / memgradclip012 / routecenter`
- 当前应把它视为“最值得继续围绕的候选结构”

### 2. `eventgateconst_quat020`：压低 quaternion residual cap（失败）

改动：

- `quat_residual_ratio_max: 0.25 -> 0.20`

**USHCN itr=10：**

- `0.1584, 0.1936, 0.2103, 0.1633, 0.1672, 0.1796, 0.1716, 0.1677, 0.1705, 0.1729`
- 均值 `0.17550`
- std `0.01473`
- max `0.21027`

**结论：**

- 数值只比 `eventgateconst` 略有改善
- 但诊断日志里 `quat_clip = 0.0`
- 说明 residual cap 基本没有真正触发
- 因此它不是根因级改动，已回退

### 3. `eventgateconst_meminit001`：给 `membrane_proj` 小随机初始化（失败）

假设：

- 坏轮可能来自 route 早期分散度起不来

改动：

- `membrane_proj.weight` 从零初始化改为 `std=0.01` 小随机初始化

**USHCN itr=10：**

- `0.2481, 0.1705, 0.1651, 0.1719, 0.4619, 0.2173, 0.1789, 0.1682, 0.1820, 0.1749`
- 均值 `0.21388`
- std `0.08635`
- max `0.46190`

**结论：**

- 明显失败
- 人为抬高早期 route dispersion 会破坏稳定性
- 该方向已经证伪，已回退

### 4. `eventgateconst_routedetach`：切断 route density 到稳定化支路的反向梯度（失败）

假设：

- 坏轮可能来自 density-aware 稳定化分支反向牵引 `route_logit`

改动：

- 在 `summarize_route_density` 中对 `route_logit` 做 `detach()`

**USHCN itr=10：**

- `0.1595, 0.1933, 0.1625, 0.1694, 0.1673, 0.2329, 0.1642, 0.1724, 0.2184, 0.1723`
- 均值 `0.18122`
- std `0.02408`
- max `0.23287`

**结论：**

- 也明显失败
- 说明 route density 虽然带来风险，但不能简单切断其训练反馈
- 该方向已回退

### 当前阶段性总判断

截至 2026-04-19，围绕 `USHCN` 尾部压制的最新结论是：

1. `eventgateconst` 是当前唯一明确值得保留的候选。
2. 直接压 quaternion residual cap 不是根因修复。
3. 直接提高 route 初始分散度会明显恶化稳定性。
4. 直接切断 route density 的反向梯度也会恶化稳定性。
5. 后续不应继续重复：
   - `membrane` 初始化改动
   - route density detach
   - 单纯 residual cap 收缩
6. 下一步如果继续做结构试验，应该把注意力放在：
   - `eventgateconst` 框架下
   - quaternion 参数演化方式本身
   - 而不是继续粗暴碰 route 前端

在不改变 M1 总体结构的前提下，曾尝试将 `retain_gate`、`event_gate` 与 `event_residual_scale` 的增长改为更保守的有界形式。

| 数据集 | 轮数 | MSE 均值 ± std | 结论 |
|--------|------|----------------|------|
| USHCN | 10 | **0.2068 ± 0.0375** | 明显退化，整套试验回退 |
| P12 | 5 | **0.3012 ± 0.0012** | 基本不变 |
| MIMIC_III | 5 | **0.3983 ± 0.0066** | 有小幅改善，但不具备统一推广性 |

**结论：** bounded-gate 不是统一主线方案，当前仓库已回到上一版 M1 主线。

### 阶段 2：本地参数诊断确认旧 M1 的真实动力学

在 `exp/exp_main.py` 中加入 epoch 级 `QSHDiag` 之后，对 `USHCN` 与 `HumanActivity` 做了短程本地诊断。

**关键观察：**

1. 旧 M1 中的 `event residual` 分支最初是死分支。
   - `event_proj.weight norm = 0.0`
   - `event_residual_scale = 0.0`
   - `event_log_scale` 长期停留在极低区间

2. 真正持续学习的是：
   - `retain_log_scale`
   - `membrane_proj.weight`
   - `quat_gate`
   - `quat_h2n`

3. USHCN 的坏轮更像是活跃分支被高方差数据放大，而不是 `event` 协同失败。

**结构级结论：**

- 旧 M1 的真实动力学更接近 `HyperIMTS + retain 调制 + quaternion 残差`。
- 如果继续沿论文叙事推进，就不能再把 `event` 当成默认已经工作的分支。

### 阶段 3：方向 B 稳定化试验

#### B1：`retaincap_main`

只对 `retain` 路径做幅度约束，不动 quaternion 与 event 结构。

| 数据集 | 轮数 | MSE 均值 ± std | 结论 |
|--------|------|----------------|------|
| HumanActivity | 3 | **0.0442 ± 0.0018** | 基本稳定 |
| USHCN | 5 | **0.1673 ± 0.0033** | 明显优于旧 M1，说明 `retain` 约束有效 |

#### B2：`retaincap_quatbound`

在 `retain cap` 基础上，再限制 quaternion residual 相对线性主干的残差幅度。

| 数据集 | 轮数 | MSE 均值 ± std | 结论 |
|--------|------|----------------|------|
| HumanActivity | 3 | **0.0426 ± 0.0009** | 安全 |
| USHCN | 5 | **0.1674 ± 0.0093** | 均值几乎不动，方差略放大 |

**结论：** quaternion residual 的简单有界化不是当前最关键的增益点。

### 阶段 4：方向 A 激活 `event` 分支

#### A1：`eventfusion_sigmoid`

只改 `event` 融合强度表达：

- `event_residual_scale` 从近零指数残差，改成 `sigmoid` 受控融合系数
- 初始化到 `sigmoid ≈ 0.1`

| 数据集 | 轮数 | MSE 均值 ± std | 结论 |
|--------|------|----------------|------|
| HumanActivity | 3 | **0.0426 ± 0.0009** | `event` 分支在简单数据上稳定活化 |
| USHCN | 5 | **0.1803 ± 0.0386** | `event` 活了，但高方差尾部明显放大 |

**关键意义：**

- `event_residual_scale` 不再是近零死残差语义
- `event_proj.weight norm` 在 HumanActivity 与 USHCN 上都持续偏离 0
- 说明 `event` 分支第一次真正参与训练

#### A2：`eventnorm_main`

在 `eventfusion_sigmoid` 基础上，只增加 `event` 支路的独立归一化：

- `temporal_event_norm`
- `variable_event_norm`

| 数据集 | 轮数 | MSE 均值 ± std | 结论 |
|--------|------|----------------|------|
| HumanActivity | 3 | **0.0430 ± 0.0013** | 仍然稳定 |
| USHCN | 5 | **0.1653 ± 0.0011** | 在保留 `event` 活性的同时显著压回方差 |

**结论：**

- `event` 支路的问题不只是“强度太弱”，还包括“缺少独立控制”
- `event` 独立归一化是当前最有价值的结构补充

#### A2 扩展验证：`eventnorm_itr10`

为了确认 `itr=5` 不是偶然现象，对 `eventnorm` 版本补做了 `USHCN itr=10`。

| 数据集 | 轮数 | MSE 均值 ± std | 最优 | 最差 | 结论 |
|--------|------|----------------|------|------|------|
| USHCN | 10 | **0.1829 ± 0.0279** | 0.1628 | 0.2353 | 长重复下仍有尾部风险，不能直接视为最终定版 |

**判断：**

- `eventnorm` 明显优于单纯增强融合强度的 `eventfusion_sigmoid`
- 但长重复下仍会出现 `iter5/7/8` 级别的坏轮
- 当前更适合把它写成「可训练、可控、但尚未彻底稳定」的阶段性版本

#### A2.0 失败分支：`eventgain_main`

尝试在 `eventnorm` 基础上，为 `event` 注入额外增加“主干条件化 gain”：

- `event_gain = sigmoid(Linear(main_hyperedge_state))`
- `updated = main_update + event_scale * event_gain * event_delta`

| 数据集 | 轮数 | MSE 均值 ± std | 结论 |
|--------|------|----------------|------|
| HumanActivity | 3 | **0.0424 ± 0.0006** | 简单数据上基本正常 |
| USHCN | 5 | **0.1974 ± 0.0633** | 明显退化，并重新出现重度坏轮 |

**判断：**

- `event` 支路的问题不是“缺少一层主干条件化控制器”
- 额外的条件化 gain 反而会让部分 run 更脆弱
- 这一方向应明确否定

#### A2.0 失败分支：`event_temporal_only`

尝试缩减传播范围，只保留 temporal `event` 注入，去掉 variable `event` 注入。

| 数据集 | 轮数 | MSE 均值 ± std | 结论 |
|--------|------|----------------|------|
| HumanActivity | 3 | **0.0422 ± 0.0005** | 简单数据上保持稳定 |
| USHCN | 5 | **0.1779 ± 0.0365** | 明显不如双支路 `eventnorm_main` |

**判断：**

- 双支路传播本身不是主要问题
- 只保留 temporal 注入并不能自动带来更高稳定性
- `eventnorm_main` 的有效性来自“独立控制 + 双支路表达”，而不是单纯删支路

#### A2.0 失败分支：`eventnorm_clip`

尝试在 `eventnorm_main` 基础上，直接对归一化后的 `event_delta` 做 `tanh` 幅度裁剪。

| 数据集 | 轮数 | MSE 均值 ± std | 结论 |
|--------|------|----------------|------|
| HumanActivity | 3 | **0.0422 ± 0.0005** | 简单数据上基本不受影响 |
| USHCN | 5 | **0.1898 ± 0.0381** | 出现明显双峰坏轮，方向失败 |

**判断：**

- 问题不在于归一化后的 `event_delta` 还残留少量大幅值
- 直接裁剪 `event_delta` 会破坏注入连续性，造成新的两极分化
- 这一步明确说明：应当控制“总注入量”，而不是粗暴裁剪“事件形状”

#### A2.1：`eventscalecap_main`

在 `eventnorm_main` 基础上，不改拓扑，只对 `event_scale` 引入温和上界：

- `event_scale = clamp(sigmoid(event_residual_scale), max=0.12)`

| 数据集 | 轮数 | MSE 均值 ± std | 结论 |
|--------|------|----------------|------|
| HumanActivity | 3 | **0.0430 ± 0.0013** | 整体保持稳定 |
| USHCN | 5 | **0.1663 ± 0.0027** | 与 `eventnorm_main` 接近，说明温和上界是安全保险丝 |

**判断：**

- 这一步没有像 `eventnorm_clip` 那样制造新的双峰坏轮
- 但在 `itr=5` 上也没有明显胜过 `eventnorm_main`
- 因此它更像是一个保守的总量控制，而不是短重复下的强增益点

#### A2.1 扩展验证：`eventscalecap_itr10`

为了确认 `event_scale` 上界是否对长重复稳定性真正有帮助，对 `USHCN` 补做了 `itr=10`。

| 数据集 | 轮数 | MSE 均值 ± std | 最优 | 最差 | 结论 |
|--------|------|----------------|------|------|------|
| USHCN | 10 | **0.1728 ± 0.0222** | 0.1610 | 0.2355 | 明显优于 `eventnorm_itr10`，但尾部坏轮仍未完全消失 |

**判断：**

- 相比 `eventnorm_itr10` 的 `0.1829 ± 0.0279`，温和上界版本均值和方差都更好
- 说明长重复下的一部分尾部风险确实来自 `event` 总注入量过大
- 但 `iter8 = 0.2355` 仍然表明该问题没有被彻底解决
- 因此当前最准确的表述应是：`eventnorm + mild event_scale cap` 是更好的三元素主线候选，但仍不是最终稳定版

### 当前综合结论

1. **如果目标只是追求最稳的短期结果，`retaincap_main` 仍然是最保守的稳定化版本。**
2. **如果目标是保住论文里的三元素统一框架，当前更合适的主线候选已经升级为 `eventnorm + mild event_scale cap`。**
3. **`eventnorm` 已经证明：**
   - `event` 不再是死分支
   - `event` 可以在 HumanActivity 与 USHCN 上真实参与训练
   - `event` 支路需要独立控制，否则会在高方差数据上放大尾部风险
4. **`event_scale` 温和上界进一步证明：**
   - `event` 的长重复尾部风险部分来自总注入量偏大
   - 对总注入量做轻量约束可以改善 `USHCN itr=10` 的均值和方差
   - 但仍不足以彻底消除坏轮
5. **当前最准确的表述不是“三分支已经稳定协同”，而是：**
   - 超图仍然是结构主干
   - 四元数仍然是主要增强分支
   - 脉冲 / event 已经成为真实可训练、可控的事件注入支路，并且在加入轻量总量约束后进一步改善了长重复稳定性，但长重复下仍有尾部不稳定性

## EQHO 开发记录（2026-04-17）

### 背景

在 `eventscalecap_main` 母体之上，尝试引入 `Event-Conditioned Quaternion Hyperedge Operator (EQHO)`，目标不是替换现有超图主干，而是让 `event` 通过 hyperedge-level quaternion refinement 真实参与四元数混合控制。

实现按 3 个阶段推进：

- `A1`：固定 real-only 模板，只增加 hyperedge-level quaternion refinement
- `A2`：保持固定模板，仅让 `event_summary` 控制 `mix_gain`
- `A3`：放开完整 `mix_coef`，允许 `event_summary` 动态控制 `r / i / j / k` mixing

### 本地筛查结果：`A3 = Full EQHO`

运行版本：

- `HumanActivity`：`QSHNet_EQHO_A3`
- `USHCN`：`QSHNet_EQHO_A3`

#### 结果

| 配置 | 数据集 | 轮数 | 结果 |
|------|--------|------|------|
| `QSHNet_EQHO_A3` | HumanActivity | 3 | **0.0418 ± 0.0002** |
| `QSHNet_EQHO_A3` | USHCN | 5 | **0.1979 ± 0.0310** |

对应单轮结果：

- HumanActivity：`0.04183`、`0.04208`、`0.04159`
- USHCN：`0.21483`、`0.17559`、`0.24942`、`0.18205`、`0.16750`

### `QSHDiag` 诊断结论

#### HumanActivity

- `EQHO-temporal coef_r` 长期保持在约 `0.597`
- `EQHO-variable coef_r` 长期保持在约 `0.593`
- `gain_mean` 基本稳定在 `0.0192 ~ 0.0195`
- `temporal residual_norm_mean` 约 `0.06`
- `variable residual_norm_mean` 大约 `0.11 ~ 0.14`

判断：

- `EQHO` 已真实参与前向，不是死结构
- 但其行为更接近温和 refinement，而不是强烈改变主导表达的分支
- `temporal / variable` 两支路有差异，但差异幅度有限

#### USHCN

- `EQHO-temporal coef_r` 快速塌到接近 `0`
- `EQHO-variable coef_r` 同样接近 `0`
- `temporal` 侧主要由 `coef_i / coef_j` 主导
- `variable` 侧长期由 `coef_k` 主导
- `gain_mean` 升到 `0.07 ~ 0.14`
- `summary_mean` 升到 `2.4 ~ 3.5`
- `residual_norm_mean` 升到 `1.4 ~ 1.8`

判断：

- `A3` 不是“没学起来”，而是学得过强
- 完整动态 `mix_coef` 在高方差数据上会把 quaternion hyperedge refinement 推到非实部主导区间
- 该机制会重新放大 `USHCN` 的坏轮风险

### 结论

1. `EQHO` 本身不是无效设计。
   - 它在 `HumanActivity` 与 `USHCN` 上都表现出真实可训练性。

2. `A3` 不能作为当前主线继续推进。
   - `HumanActivity` 稳定且结果不差。
   - 但 `USHCN itr=5` 明显退化到 `0.1979 ± 0.0310`，远差于当前主线候选 `eventscalecap_main` 的 `0.1663 ± 0.0027`。

3. 最值得保留的不是 `A3` 本身，而是新的结构判断：
   - `event-conditioned hyperedge refinement` 有研究价值；
   - 但 `full dynamic r / i / j / k mixing` 对高方差数据过强；
   - 后续若继续做 `EQHO`，应优先回到受限动态化，而不是保留当前 `A3`。

### 当前决策更新

- `eventscalecap_main / eventscalecap_itr10` 继续作为当前统一主线候选
- `EQHO A3` 记为“可训练但失败的探索分支”
- 后续若继续推进 `EQHO`，下一步应优先尝试 `A2.5`
  - 保持 real-dominant 模板
  - 只允许 `mix_coef` 在安全边界内做小幅偏移
  - 不再直接放开完整动态 `mix_coef`

### 本地筛查结果：`A2.5 = Template-Offset + Safety Floor`

在 `A3` 失败后，继续在 `eventscalecap_main` 母体上尝试更保守的受限动态化版本：

- 使用固定 real-dominant 基模板 `mix_coef = [0.70, 0.10, 0.10, 0.10]`
- 仅允许 `event_summary` 预测有界 offset
- 对 `coef_r` 显式设置 safety floor：`coef_r >= 0.55`
- `mix_gain` 路径保持不变

运行版本：

- `HumanActivity`：`QSHNet_EQHO_A25`
- `USHCN`：`QSHNet_EQHO_A25`

#### 结果

| 配置 | 数据集 | 轮数 | 结果 |
|------|--------|------|------|
| `QSHNet_EQHO_A25` | HumanActivity | 3 | **0.0418 ± 0.0002** |
| `QSHNet_EQHO_A25` | USHCN | 5 | **0.2062 ± 0.0264** |

对应单轮结果：

- HumanActivity：`0.04178`、`0.04207`、`0.04159`
- USHCN：`0.23422`、`0.17866`、`0.22763`、`0.21162`、`0.17868`

### `A2.5` 诊断结论

#### HumanActivity

- `EQHO-temporal coef_r` 长期稳定在约 `0.699`
- `EQHO-variable coef_r` 也保持 real-dominant
- `gain_mean` 基本维持在 `0.0192 ~ 0.0196`

判断：

- `A2.5` 没有破坏简单数据上的可训练性
- 说明“real-dominant 模板 + 有界 offset”本身是安全可运行的

#### USHCN

- temporal `coef_r` 被稳定压在 `0.55`
- `coef_i / coef_j / coef_k` 被稳定限制在约 `0.15`
- `A3` 中 `mix_coef` 偏离安全区的问题被显式消除
- 但 temporal `gain_mean` 仍长期维持在 `0.14 ~ 0.15`
- `summary_mean` 仍在 `3+`，个别轮次更高

判断：

- `A2.5` 已经证明 `A3` 的主要灾难确实包含 `mix_coef` 失控这一因素
- 但它没有解决 `mix_gain` 在高方差数据上的放大问题
- 也就是说，`EQHO` 在 `USHCN` 上的主风险已经从“系数漂移”转移为“增益饱和”

### `A2.5` 的结构意义

1. `A2.5` 不是有效候选主线。
   - `USHCN itr=5` 仍退化到 `0.2062 ± 0.0264`；
   - 不仅差于 `eventscalecap_main` 的 `0.1663 ± 0.0027`，也比 `A3` 的 `0.1979 ± 0.0310` 更差。

2. 但它提供了比 `A3` 更强的结构归因证据。
   - `mix_coef` 被锁回安全区后，坏轮仍然存在；
   - 因此 `EQHO` 在 `USHCN` 上的根本风险不能再简单归因于 quaternion mixing 系数漂移。

3. 当前更准确的判断是：
   - `A3` 失败来自 `mix_coef` 与 `mix_gain` 的联合作用；
   - `A2.5` 进一步证明，即使压住 `mix_coef`，只要 `mix_gain` 仍可在 temporal 支路持续接近饱和，`USHCN` 仍会出现显著坏轮。

### `A2.5` 后的决策更新

- `A2.5` 记为“结构归因成功，但实验结果失败”的探索分支
- 后续若继续推进 `EQHO`，下一步不应再围绕 `mix_coef` 做文章
- 更值得验证的唯一结构假设将转向：
  - 在保持 `A2.5` real-dominant 安全模板不动的前提下，
  - 对 `mix_gain` 做单因素约束或重参数化，
  - 检查 `USHCN` 的坏轮是否能进一步被压回

### 本地筛查结果：`A2.6 = A2.5 + gain hard cap`

在 `A2.5` 之后，继续只改一个核心因素：将 `mix_gain` 上界从 `0.15` 压到 `0.08`，其余全部保持不变：

- `mix_coef` 仍使用 `[0.70, 0.10, 0.10, 0.10]`
- `coef_r >= 0.55` 的 safety floor 不变
- `event_summary`、`coef_head`、quaternion residual 主体均不改

运行版本：

- `HumanActivity`：`QSHNet_EQHO_A26`
- `USHCN`：`QSHNet_EQHO_A26`

#### 结果

| 配置 | 数据集 | 轮数 | 结果 |
|------|--------|------|------|
| `QSHNet_EQHO_A26` | HumanActivity | 3 | **0.0418 ± 0.0003** |
| `QSHNet_EQHO_A26` | USHCN | 5 | **0.1846 ± 0.0297** |

对应单轮结果：

- HumanActivity：`0.04177`、`0.04207`、`0.04153`
- USHCN：`0.22667`、`0.16272`、`0.20463`、`0.17026`、`0.15859`

### `A2.6` 诊断结论

#### HumanActivity

- temporal / variable `gain_mean` 都稳定在 `0.0103 ~ 0.0104`
- `coef_r` 继续保持在约 `0.699`
- 结果与 `A2.5` 基本持平

判断：

- `gain hard cap` 没有把简单数据上的 EQHO 表达压死
- 说明把 `mix_gain` 上界直接降到 `0.08` 在小数据上仍是安全的

#### USHCN

- temporal `gain_mean` 被稳定压到 `0.075 ~ 0.080`
- 不再出现 `A2.5` 中 `0.14 ~ 0.15` 的持续饱和
- `coef_r` 继续被锁在 `0.55` 左右
- 但 temporal `summary_mean` 仍然长期在 `5 ~ 11`
- `residual_norm_mean` 仍然长期在 `1.63 ~ 1.76`

判断：

- `A2.6` 已经证明：单独压 `mix_gain` 上界，确实可以改善 `USHCN`
- 但它没有从根本上消除高能量 `event_summary` 驱动的放大问题

### `A2.6` 的结构意义

1. `A2.6` 是当前 `EQHO` 探索里最好的受限版本。
   - 相比 `A2.5` 的 `0.2062 ± 0.0264`，改善到 `0.1846 ± 0.0297`
   - 相比 `A3` 的 `0.1979 ± 0.0310`，也有实质改善

2. 但 `A2.6` 仍然不能进入主线。
   - 它仍明显差于 `eventscalecap_main` 的 `0.1663 ± 0.0027`
   - 并且方差仍明显偏大

3. 当前关于 `EQHO` 的最准确判断进一步更新为：
   - `mix_coef` 漂移不是唯一问题；
   - `mix_gain` 饱和也确实是问题；
   - 但即使同时压住 `mix_coef` 和 `mix_gain` 的显式上界，`event_summary` 驱动的高方差放大仍然存在。

### `A2.6` 后的决策更新

- `A2.6` 记为“比 `A2.5/A3` 更好，但仍未达到主线标准”的探索分支
- 如果后续继续推进 `EQHO`，不应再重复单纯的 `mix_coef` 边界修补
- 也不应再重复只做固定 `mix_gain` 上界压缩
- 下一轮若继续，需要转向更深一层的问题：
  - `event_summary` 的幅值与统计形态本身
  - 而不仅仅是它之后的 `coef` 或 `gain` 投影

### 当前决策

- **保留 `eventnorm + mild event_scale cap` 作为当前三元素统一框架候选版本。**
- **不再回到常规超参数扫描。**
- **后续若继续改动，应优先围绕 `event_scale` 的工作区间与更细的尾部稳定性做单因素控制，而不是继续无差别增强 `event` 强度。**

### 本地筛查结果：`S1 = A2.6 + event_summary output LayerNorm`

在 `A2.6` 之后，尝试只改一个更上游的核心因素：

- 在 `HyperedgeEventSummarizer` 输出端加入 `LayerNorm(cond_dim)`
- 其余保持 `A2.6` 不变：
  - `mix_coef` 仍为 `[0.70, 0.10, 0.10, 0.10]`
  - `coef_r >= 0.55`
  - `mix_gain_max = 0.08`

运行版本：

- `HumanActivity`：`QSHNet_EQHO_S1`
- `USHCN`：`QSHNet_EQHO_S1`

#### 结果

| 配置 | 数据集 | 轮数 | 结果 |
|------|--------|------|------|
| `QSHNet_EQHO_S1` | HumanActivity | 3 | **0.0418 ± 0.0003** |
| `QSHNet_EQHO_S1` | USHCN | 5 | **0.2072 ± 0.0461** |

对应单轮结果：

- HumanActivity：`0.04181`、`0.04206`、`0.04152`
- USHCN：`0.18965`、`0.28756`、`0.20210`、`0.18195`、`0.17460`

### `S1` 诊断结论

#### HumanActivity

- 结果与 `A2.6` 基本持平
- 说明 `event_summary` 输出端的 `LayerNorm` 不会破坏简单数据上的可训练性

#### USHCN

- 首轮 `QSHDiag` 已显示：
  - temporal `summary_mean ≈ 0.20`
  - variable `summary_mean ≈ 0.22`
- 相比 `A2.6` 在 `USHCN` 上长期出现的 `summary_mean = 5 ~ 11`，统计量被显著压低
- 但最终 `USHCN itr=5` 反而退化到 `0.2072 ± 0.0461`
- 并且方差比 `A2.6` 更大，出现了 `0.28756` 的明显坏轮

判断：

- `event_summary` 输出归一化确实改变了统计尺度
- 但这并没有转化为更好的 `USHCN` 最终表现
- 说明问题不能被简单表述为“summary 幅值过大，所以只要直接归一化输出就能解决”

### `S1` 的结构意义

1. `event_summary` 的统计量下降，不等于 `USHCN` 性能改善。
   - `S1` 把 `summary_mean` 从高能区压回了低量级；
   - 但最终 MSE 仍明显差于 `A2.6` 与主线。

2. 因此 “只在 summarizer 输出端加 `LayerNorm`” 不是正确修复方向。
   - 它更像是把表面统计压平了；
   - 但没有修复真正决定泛化效果的结构问题。

3. 当前关于 `EQHO` 的判断需要再更新一层：
   - `mix_coef` 不是主因；
   - `mix_gain` 不是充分解释；
   - `event_summary` 输出尺度本身也不是唯一主因；
   - 更深的问题可能在 `event_summary` 的信息组织方式，而不是单纯幅值。

### `S1` 后的决策更新

- `S1` 记为“统计上压住了 `summary`，但实验结果失败”的探索分支
- 后续若继续推进 `EQHO`，不应再沿着“只做 summarizer 输出归一化”继续细调
- `EQHO` 当前仍然不能替代 `eventscalecap_main`

### 本地筛查结果：`S2 = structured branch fusion`

在 `S1` 之后，继续只改 summarizer 内部融合方式：

- 不再使用 `cat + fuse`
- 改成分路投影后做静态加权聚合
- 其余仍保持 `A2.6` 的 `mix_coef / gain_max` 约束

运行版本：

- `HumanActivity`：`QSHNet_EQHO_S2`
- `USHCN`：`QSHNet_EQHO_S2`

#### 结果

| 配置 | 数据集 | 轮数 | 结果 |
|------|--------|------|------|
| `QSHNet_EQHO_S2` | HumanActivity | 3 | **0.0422 ± 0.0002** |
| `QSHNet_EQHO_S2` | USHCN | 5 | **0.2218 ± 0.0336** |

对应单轮结果：

- HumanActivity：`0.04206`、`0.04245`、`0.04212`
- USHCN：`0.20329`、`0.25454`、`0.22815`、`0.24934`、`0.17376`

### `S2` 诊断结论

- `HumanActivity` 已经出现轻度退化
- `USHCN` 则进一步恶化到比 `S1` 更差的水平
- 首轮 `QSHDiag` 中，`coef_r` 甚至在 variable 支路掉到 `0.56 ~ 0.61` 附近

判断：

- 这种“显式三路加权聚合”虽然改变了信息组织方式
- 但它破坏了 `EQHO` 需要的 real-dominant 安全结构
- 因而是明确失败方向

### 本地筛查结果：`S3 = main summary + bounded residual event/gate`

在 `S2` 失败后，进一步尝试更保守的 summarizer 结构：

- `summary = main_feat + alpha * event_feat + beta * gate_feat`
- 其中 `alpha / beta` 是有界残差系数
- 目标是让 summarizer 本身也遵循“主干主导、事件小残差修正”的主线范式

运行版本：

- `HumanActivity`：`QSHNet_EQHO_S3`
- `USHCN`：`QSHNet_EQHO_S3`

#### 结果

| 配置 | 数据集 | 轮数 | 结果 |
|------|--------|------|------|
| `QSHNet_EQHO_S3` | HumanActivity | 3 | **0.0418 ± 0.0004** |
| `QSHNet_EQHO_S3` | USHCN | 5 | **0.2173 ± 0.0340** |

对应单轮结果：

- HumanActivity：`0.04145`、`0.04219`、`0.04164`
- USHCN：`0.24181`、`0.24650`、`0.18545`、`0.23759`、`0.17537`

### `S3` 诊断结论

- `HumanActivity` 回到了安全线附近，说明 residual-style summarizer 至少没有破坏简单数据
- 但 `USHCN` 仍明显退化，且远差于 `A2.6`
- 尽管 temporal 分支较稳，variable 分支的 `coef_r` 仍会逼近安全下界

判断：

- 把 summarizer 改成 residual-style，仍不足以修复 `USHCN`
- 说明问题并不只是 summarizer 的融合拓扑
- 至少在当前 `EQHO` 设计里，继续围绕 summarizer 小修小补已经没有性价比

### `S2/S3` 后的决策更新

- `S2` 与 `S3` 一起给出更强的否定性结论：
  - 仅围绕 `event_summary` 的输出尺度或内部融合拓扑做改造，不能把 `EQHO` 拉回主线水位
- 因此 `EQHO` 当前最优版本仍然是 `A2.6`
- 但 `A2.6` 依旧明显弱于 `eventscalecap_main`
- 后续不应继续在 summarizer 层做局部结构修补

## `event density` 试验链（2026-04-18）

### 背景

在 `eventscalecap_main` 母体之上，继续只看 `event` 尾部控制，尝试回答一个更窄的问题：

- 如果 `USHCN` 的坏轮来自某些高活跃 route 把真实有效的 `event` 注入放大，
- 那么是否能通过更局部的 `event` 收缩，把坏轮压回去，同时保住 `HumanActivity` 上已经形成的改善？

这轮试验严格遵守单因素原则，只沿着 `event residual / event density` 这条线推进。

### 试验 1：`eventrescap_main`

设计：

- 不改变 `eventnorm + eventscalecap` 的基本结构
- 仅额外约束 `event_scale * event_delta` 的总残差范数
- 将其相对 `main_state` 范数限制在固定比例以内

#### 结果

| 配置 | 数据集 | 轮数 | 结果 |
|------|--------|------|------|
| `eventrescap_main` | HumanActivity | 3 | **0.04185 ± 0.00009** |
| `eventrescap_main` | USHCN | 5 | **0.1834 ± 0.0273** |

对应单轮结果：

- HumanActivity：`0.04181`、`0.04195`、`0.04178`
- USHCN：`0.15558`、`0.22176`、`0.20143`、`0.17132`、`0.16689`

#### 结论

- `HumanActivity` 继续保持改善
- 但 `USHCN` 的均值和方差都明显退化
- 说明“按主干范数对 event residual 做比例硬约束”会把高方差数据上的有效 `event` 一起压掉

因此：

- `eventrescap_main` 记为失败方向，不保留

### 试验 2：`eventdenscap_main`

设计：

- 不再直接约束 residual 范数
- 改为根据 route density 动态衰减 `event_scale`
- 且对 temporal / variable 两条路径同时生效

#### 结果

| 配置 | 数据集 | 轮数 | 结果 |
|------|--------|------|------|
| `eventdenscap_main` | HumanActivity | 3 | **0.04181 ± 0.00011** |
| `eventdenscap_main` | USHCN | 5 | **0.1891 ± 0.0347** |

对应单轮结果：

- HumanActivity：`0.04182`、`0.04191`、`0.04169`
- USHCN：`0.18448`、`0.24986`、`0.17584`、`0.16747`、`0.16795`

#### 结论

- `HumanActivity` 仍然不受伤
- 但 `USHCN` 进一步退化，甚至比 `eventrescap_main` 更差
- 说明“全路径 density-aware 抑制”不是安全保险丝
- 更准确地说，temporal 路径上的 density-aware 收缩本身就会误伤有效注入

因此：

- `eventdenscap_main` 记为失败方向，不保留

### 试验 3：`eventdensvar_main`

设计：

- 保持 temporal 路径与 `eventscalecap_main` 完全一致
- 只在 variable 路径上保留 density-aware `event_scale` 衰减
- 目标是只对更可能放大局部噪声的 variable 注入做温和控制

#### 结果

| 配置 | 数据集 | 轮数 | 结果 |
|------|--------|------|------|
| `eventdensvar_main` | HumanActivity | 3 | **0.04181 ± 0.00011** |
| `eventdensvar_main` | USHCN | 5 | **0.1703 ± 0.0058** |

对应单轮结果：

- HumanActivity：`0.04176`、`0.04194`、`0.04174`
- USHCN：`0.17625`、`0.16607`、`0.17673`、`0.16748`、`0.16481`

#### 结论

- `HumanActivity` 的改善被完整保住
- `USHCN` 相比 `eventscalecap_main` 的 `0.1663 ± 0.0027`，均值略差
- 但它显著好于 `eventrescap_main` 与 `eventdenscap_main`
- 同时方差已经收敛到 `0.0058` 量级，没有再出现严重长尾

因此当前更准确的定位是：

- `eventdensvar_main` 不是新的统一主线
- 但它已经达到“结果可接受、值得保留”的状态
- 若后续继续沿 `event density` 方向推进，应以它为直接起点，而不是回到更激进的全路径收缩

### 阶段 5：服务器四数据集正式验证

在本地筛选确认 `eventdensvar_main` 可保留之后，进一步对它做了服务器四数据集验证：

| 配置 | 数据集 | 轮数 | 结果 |
|------|--------|------|------|
| `eventdensvar_main` | HumanActivity | 5 | **0.04174 ± 0.00019** |
| `eventdensvar_main` | USHCN | 10 | **0.1886 ± 0.0324** |
| `eventdensvar_main` | P12 | 5 | **0.30092 ± 0.00062** |
| `eventdensvar_main` | MIMIC_III | 5 | **0.39791 ± 0.01530** |

对应单轮结果：

- HumanActivity：`0.041741`、`0.041906`、`0.041728`、`0.041872`、`0.041431`
- USHCN：`0.159863`、`0.167083`、`0.223942`、`0.162037`、`0.165167`、`0.209938`、`0.162052`、`0.168732`、`0.226081`、`0.240721`
- P12：`0.301369`、`0.300219`、`0.301650`、`0.300377`、`0.300964`
- MIMIC_III：`0.391491`、`0.425247`、`0.389874`、`0.391132`、`0.391809`

#### 服务器验证结论

- `HumanActivity` 上的改善被再次确认，而且稳定复现
- `P12` 结果稳定，通过跨数据集可用性检查
- `MIMIC_III` 均值保持在可接受区间，但仍存在单轮失稳
- `USHCN` 在 `itr=10` 下重新出现明显坏轮，说明本地 `itr=5` 的“可接受”并不能外推为正式稳定结论

因此这一步之后，`eventdensvar_main` 的最终定位需要进一步收紧为：

- 它仍然是当前 `event density` 方向唯一值得保留的候选
- 但它不能升级为新的统一主线
- 当前统一主线仍应保持为 `eventscalecap_main / eventscalecap_itr10`
- 后续若继续沿这条线推进，唯一合理目标是压制 `USHCN` 坏轮，同时不能破坏 `HumanActivity / P12` 的已确认收益

### 这轮试验链带来的新增约束

1. 不要再做全局 `event residual` 比例硬约束。
2. 不要再同时对 temporal / variable 两条路径做 density-aware 收缩。
3. 若继续推进 `event density` 方向，只看 variable 路径。
4. `eventscalecap_main / eventscalecap_itr10` 仍是统一主线母体；
   `eventdensvar_main` 则是当前保留候选，而不是新的统一主线。
5. 后续若继续做结构优化，优先目标不再是进一步改善 `HumanActivity`，
   而是只针对 `USHCN` 坏轮做抑制，并同时守住 `P12 / MIMIC_III` 的可接受区间。

## 2026-04-18：`variable residual only` 后的长重复坏轮诊断准备

在 `USHCN itr=5` 达到可接受结果后，继续做了 `USHCN itr=10` 长重复检查。

### 当前接受主线

当前工作区默认保留的是：

- `adaptive fused-cap`
- `variable residual only` fused-context 稳定化
- 不启用 quaternion risk ceiling
- 不启用 retain risk ceiling

该结构的核心含义是：

- temporal context 继续保持原路径，不做 fused residual 截断；
- variable context 只保留相对 base context 的有界残差；
- fused route density 取 temporal 与 variable 两条路径的最大值，用于自适应收紧 variable residual cap；
- 事件注入、retain 调制与 quaternion residual 都保持可训练，但不再额外叠加单点 ceiling。

### 已确认结果

`variable residual only` 在 `USHCN itr=5` 上的结果：

| 数据集 | 轮数 | MSE 均值 ± std | 单轮最大值 | 结论 |
|------|------|----------------|------------|------|
| `USHCN` | 5 | `0.16785 ± 0.00772` | `0.17962` | 用户接受，作为当前可用主线 |

对应单轮 MSE：

- `0.159564`
- `0.164314`
- `0.170925`
- `0.179617`
- `0.164842`

同一结构在 `USHCN itr=10` 上的结果：

| 数据集 | 轮数 | MSE 均值 ± std | 单轮最大值 | 结论 |
|------|------|----------------|------------|------|
| `USHCN` | 10 | `0.18516 ± 0.02619` | `0.23737` | 长重复下仍有坏尾 |

对应单轮 MSE：

- `0.173720`
- `0.163736`
- `0.184923`
- `0.167952`
- `0.164524`
- `0.173355`
- `0.163550`
- `0.237369`
- `0.202104`
- `0.220376`

### 已失败的单点 ceiling 试验

#### quaternion risk ceiling

- `USHCN itr=10`: `0.18779 ± 0.03152`
- 单轮最大值：`0.24483`
- 结论：没有压低坏轮上界，反而略差，已撤回。

#### retain risk ceiling

- `USHCN itr=10`: `0.18539 ± 0.02794`
- 单轮最大值：`0.24303`
- 结论：没有解决坏尾，已撤回。

### 当前诊断判断

坏轮不应再简单归因于某一个 gate 过大。更合理的判断是：

- 坏轮来自训练轨迹差异，而不是前几轮污染后几轮；
- 每个 `itr` 使用独立 seed 与独立模型初始化；
- 高风险 seed 下，retain、variable fused residual 与 quaternion residual 的组合更容易进入放大轨迹；
- 单独压 quaternion 或单独压 retain 都不足以解决问题。

### 本次新增诊断能力

为避免继续猜测式结构修改，当前代码新增了更细粒度的 `QSHDiag` 轨迹日志。

每个 epoch 现在会记录：

- `train_loss / vali_loss / lr`
- `retain_mean / retain_min / retain_std`
- `event_mean / event_max`
- `route_logit_mean / route_logit_std`
- `temporal_density_mean / variable_density_mean`
- `temporal_ratio_mean / variable_ratio_mean`
- `fused_clip / fused_ratio_mean / fused_cap_mean`
- `quat_alpha_mean / quat_alpha_max`
- `quat_raw_ratio_mean / quat_bound_ratio_mean / quat_bound_ratio_max / quat_clip`
- 核心梯度范数：`retain_log_scale`、`membrane_proj`、`event_proj`、`event_residual_scale`、`quat_gate`、`quat_h2n`

同时新增解析脚本：

```bash
python scripts/QSHNet/extract_qshdiag.py train.log -o qshdiag.csv
```

后续判断 `USHCN itr=10` 坏轮时，应优先比较好轮与坏轮在早期 epoch 的这些字段，而不是继续直接添加新的 gate。

## 2026-04-18：`USHCN itr=10` 轨迹诊断与 route-bound 失败试验

在新增 `QSHDiag` 后，重新对当前 `variable residual only + adaptive fused-cap` 主线做了一轮本地 `USHCN itr=10` 诊断运行。

运行版本：

- `model_id = coupledctxadapt_diag_itr10`
- 日志：`storage/logs/qshdiag/ushcn_coupledctxadapt_diag_itr10_20260418_230326.log`
- 解析 CSV：`storage/logs/qshdiag/ushcn_coupledctxadapt_diag_itr10_20260418_230326_qshdiag.csv`

### 诊断运行结果

| 数据集 | 轮数 | MSE 均值 ± std | 单轮最大值 | 结论 |
|------|------|----------------|------------|------|
| `USHCN` | 10 | `0.19035 ± 0.03078` | `0.25503` | 坏轮仍存在 |

单轮 MSE：

- `iter0`: `0.172467`
- `iter1`: `0.173562`
- `iter2`: `0.176772`
- `iter3`: `0.169073`
- `iter4`: `0.164354`
- `iter5`: `0.172374`
- `iter6`: `0.168195`
- `iter7`: `0.255032`
- `iter8`: `0.230167`
- `iter9`: `0.221551`

### 诊断发现

用 `metric.json` 将 `MSE >= 0.20` 的 iteration 标为坏轮后，对比好轮和坏轮早期 epoch，主要差异集中在：

- `L0_route_logit_std`
- `L0_membrane_w_grad`

代表性现象：

- epoch 4：坏轮 `L0_route_logit_std` 高于好轮约 `+0.1746`
- epoch 5：坏轮 `L0_route_logit_std` 高于好轮约 `+0.2696`
- epoch 4：坏轮 `L0_membrane_w_grad` 高于好轮约 `+0.9577`

而以下字段不是最早、最稳定的异常来源：

- `quat_alpha_mean`
- `quat_bound_ratio_max`
- `fused_clip`

这说明坏轮并不是 quaternion residual 单独过强，也不是 fused residual cap 单独失效；更像是 spike route 轨迹进入了不同状态。

### 单因素试验：`routebound075`

基于上述诊断，尝试只改一个核心因素：

- 在 `SpikeRouter` 内对 `route_logit` 加 `tanh` 软边界
- 边界：`abs(route_logit) <= 0.75`
- 其他结构不变

运行版本：

- `model_id = routebound075_itr10`
- 日志：`storage/logs/qshdiag/ushcn_routebound075_itr10_20260418_232439.log`
- 解析 CSV：`storage/logs/qshdiag/ushcn_routebound075_itr10_20260418_232439_qshdiag.csv`

结果：

| 数据集 | 轮数 | MSE 均值 ± std | 单轮最大值 | 结论 |
|------|------|----------------|------------|------|
| `USHCN` | 10 | `0.18132 ± 0.02602` | `0.25184` | 均值略降，但坏轮上界未解决 |

单轮 MSE：

- `iter0`: `0.191665`
- `iter1`: `0.166341`
- `iter2`: `0.196957`
- `iter3`: `0.163167`
- `iter4`: `0.166984`
- `iter5`: `0.168853`
- `iter6`: `0.163077`
- `iter7`: `0.251841`
- `iter8`: `0.175908`
- `iter9`: `0.168407`

### `routebound075` 结论

- `routebound075` 压低了原本部分后段坏轮，但没有压住 `iter7`。
- 单轮最大值 `0.25184` 高于当前可接受目标 `<= 0.18`，也高于前一轮诊断运行的坏轮上界。
- 该试验不能进入主线，代码已撤回。

更重要的是，`routebound075` 让坏轮的主要差异从 route dispersion 转移到 `event_proj_w_norm` 偏低：

- 早期/中期 epoch 中，坏轮 `L0_event_proj_w_norm` 持续低于好轮约 `0.7 ~ 0.8`

这说明：

1. 单纯压 route logit 分布宽度不是充分修复。
2. 坏轮可能与 event projection 是否充分学起来有关。
3. 后续如果继续结构优化，应考虑“让 event projection 学得更稳”或“避免 event projection 学不足时影响主干”的机制，而不是继续压 `route_logit`、`retain_gate` 或 `quat_gate`。

## 2026-04-19：`eventprojgrad2` 失败试验

在 `routebound075` 失败后，进一步检查原主线与 `routebound075` 的坏轮轨迹，发现坏轮普遍伴随 `L0_event_proj_w_norm` 偏低：

- 原主线诊断运行中，坏轮 `event_proj_w_norm` 在早期 epoch 低于好轮约 `0.15 ~ 0.30`
- `routebound075` 中，这个差距扩大到约 `0.7 ~ 0.8`

因此尝试只改一个训练动力学因素：

- 不改前向结构；
- 不改 `route_logit`、`retain_gate`、`quat_gate`；
- 只给 `SpikeRouter.event_proj.weight/bias` 注册梯度 hook；
- 将 `event_proj` 的梯度放大 `2x`。

运行版本：

- `model_id = eventprojgrad2_itr10`
- 日志：`storage/logs/qshdiag/ushcn_eventprojgrad2_itr10_20260419_085215.log`
- 解析 CSV：`storage/logs/qshdiag/ushcn_eventprojgrad2_itr10_20260419_085215_qshdiag.csv`

结果：

| 数据集 | 轮数 | MSE 均值 ± std | 单轮最大值 | 结论 |
|------|------|----------------|------------|------|
| `USHCN` | 10 | `0.18962 ± 0.03279` | `0.26653` | 坏轮更差，失败 |

单轮 MSE：

- `iter0`: `0.181543`
- `iter1`: `0.165239`
- `iter2`: `0.176374`
- `iter3`: `0.170394`
- `iter4`: `0.162720`
- `iter5`: `0.172497`
- `iter6`: `0.161393`
- `iter7`: `0.266528`
- `iter8`: `0.220469`
- `iter9`: `0.219089`

### 结论

`eventprojgrad2` 没有解决坏轮，反而把最大坏轮推高到 `0.26653`。

这说明：

1. `event_proj_w_norm` 偏低确实是坏轮相关信号，但不能简单理解为“让 event_proj 学得更快就会更稳”。
2. 坏轮更可能来自 event route / event projection / fused context 之间的耦合轨迹，而不是单一参数学习速度不足。
3. 继续做单点放大、单点 ceiling、单点梯度倍率都不合适。
4. 后续更合理的结构方向应转向“事件支路置信度/可用性控制”：当 event projection 没有形成稳定表征时，降低其对主干上下文的有效影响；当其稳定后再允许参与。

该试验代码已撤回，不进入主线。

## 2026-04-19：`routeconfvar` 失败试验

在 `eventprojgrad2` 失败后，进一步尝试“事件支路置信度/可用性控制”的最小版本。

设计目标：

- 不改 `retain_gate`、`quat_gate`、`route_logit` 本身；
- 不再加速 `event_proj` 学习；
- 只在 `variable` event residual 注入处加入 route dispersion 感知衰减；
- 当 batch 内 `route_logit.std` 偏高时，将 variable event 注入强度降到最低 `0.7x`；
- temporal event 路径保持不变。

运行版本：

- `model_id = routeconfvar_itr10`
- 日志：`storage/logs/qshdiag/ushcn_routeconfvar_itr10_20260419_091933.log`
- 解析 CSV：`storage/logs/qshdiag/ushcn_routeconfvar_itr10_20260419_091933_qshdiag.csv`

结果：

| 数据集 | 轮数 | MSE 均值 ± std | 单轮最大值 | 结论 |
|------|------|----------------|------------|------|
| `USHCN` | 10 | `0.19155 ± 0.03720` | `0.26306` | 后段坏轮更差，失败 |

单轮 MSE：

- `iter0`: `0.154662`
- `iter1`: `0.164823`
- `iter2`: `0.170017`
- `iter3`: `0.162109`
- `iter4`: `0.163685`
- `iter5`: `0.162282`
- `iter6`: `0.214549`
- `iter7`: `0.263061`
- `iter8`: `0.220784`
- `iter9`: `0.239500`

早期 `QSHDiag` 对比显示，坏轮仍然主要伴随：

- `L0_membrane_w_grad` 偏高；
- `L0_route_logit_std` 偏高；
- `L0_event_proj_w_norm` 略低；
- 新增的 `variable_route_stability` 确实在坏轮中降低，但没有阻断坏轮。

### 结论

`routeconfvar` 说明：仅根据 route dispersion 衰减 variable event residual 不是有效稳定化手段。

这进一步排除了一个方向：

- 不应继续做“看到 route 不稳定就直接缩 event 注入”的局部规则；
- 坏轮不是单纯由 event residual 注入强度过大造成；
- `membrane_proj` 的早期梯度轨迹仍然是更核心的风险信号。

该试验代码已撤回，不进入主线。当前代码仍回到 `variable residual only + adaptive fused-cap` 主线。

## 2026-04-19：`routecenter` 失败试验

在 `memgradclip012` 失败后，进一步尝试一个更上层的 route 动力学控制：

- 不裁剪梯度；
- 不做 route logit 幅值 bound；
- 不改 retain / event / quaternion；
- 只把 `membrane` 输出在每个样本内做均值中心化；
- 使 route logit 表达“相对异常程度”，避免某些 seed 早期整体 route density 漂移。

运行版本：

- `model_id = routecenter_itr10`
- 日志：`storage/logs/qshdiag/ushcn_routecenter_itr10_20260419_101545.log`
- 解析 CSV：`storage/logs/qshdiag/ushcn_routecenter_itr10_20260419_101545_qshdiag.csv`

结果：

| 数据集 | 轮数 | MSE 均值 ± std | 单轮最大值 | 结论 |
|------|------|----------------|------------|------|
| `USHCN` | 10 | `0.18433 ± 0.03669` | `0.27430` | 坏轮更差，失败 |

单轮 MSE：

- `iter0`: `0.160960`
- `iter1`: `0.274295`
- `iter2`: `0.162089`
- `iter3`: `0.167135`
- `iter4`: `0.164695`
- `iter5`: `0.172360`
- `iter6`: `0.163384`
- `iter7`: `0.170591`
- `iter8`: `0.236354`
- `iter9`: `0.171457`

早期 `QSHDiag` 对比显示：

- route mean 被中心化后，坏轮仍然存在；
- 坏轮 `L0_route_logit_std` 仍高于好轮；
- `epoch=6` 坏轮 `route_logit_std` 比好轮高约 `+0.2324`；
- 坏轮 `L0_event_proj_w_norm` 仍偏低；
- `iter1 = 0.27430` 说明均值中心化可能引入新的早期不稳定。

### 结论

`routecenter` 说明：坏轮不是简单的 route mean 漂移问题。

这进一步排除：

- 直接对 route logit 做均值中心化；
- 继续围绕 route 的一阶统计量做局部修补；
- 把 USHCN 坏轮简单归因于 route density 整体偏高。

该试验代码已撤回，不进入主线。当前代码仍回到 `variable residual only + adaptive fused-cap` 主线。

## 2026-04-19：`spikeselectprop_a1` 传播选择版 spike 试验（失败）

在 `eventgateconst` 成为当前唯一值得保留的尾部压制候选后，进一步从论文叙事角度尝试把 spike 从「event 幅值控制器」改成「传播选择器」：

- 保留 `eventgateconst`；
- 不改 `event_residual_scale`；
- 不改 fused-cap；
- 不改 quaternion refinement；
- 新增 `selection_weight = sigmoid(route_logit)`；
- 在 node-to-hyperedge 前把 `obs_base` 改为 `obs_selected = obs_base * selection_weight`；
- 即让 spike 显式控制 observation 对 temporal / variable 超图传播的参与强度。

运行版本：

- `model_id = spikeselectprop_a1_itr10`
- 日志：`storage/logs/qshdiag/ushcn_spikeselectprop_a1_itr10.log`
- 解析 CSV：`storage/logs/qshdiag/ushcn_spikeselectprop_a1_itr10_qshdiag.csv`

结果：

| 数据集 | 轮数 | MSE 均值 ± std | 单轮最大值 | 结论 |
|------|------|----------------|------------|------|
| `USHCN` | 10 | `0.19942 ± 0.02713` | `0.26434` | 明显退化，失败 |

单轮 MSE：

- `iter0`: `0.207281`
- `iter1`: `0.204167`
- `iter2`: `0.193034`
- `iter3`: `0.182432`
- `iter4`: `0.227393`
- `iter5`: `0.177274`
- `iter6`: `0.169384`
- `iter7`: `0.191883`
- `iter8`: `0.264342`
- `iter9`: `0.177034`

与 `eventgateconst` 对比：

| 版本 | USHCN itr=10 MSE | std | max | `>0.18` 轮数 | `>=0.20` 轮数 |
|------|------------------|-----|-----|--------------|---------------|
| `eventgateconst` | `0.17587` | `0.01571` | `0.21558` | 3 | 1 |
| `spikeselectprop_a1` | `0.19942` | `0.02713` | `0.26434` | 7 | 4 |

### 结论

`spikeselectprop_a1` 证明：把 `sigmoid(route_logit)` 直接乘到 node-to-hyperedge observation message 上，虽然叙事更像「spike-driven propagation selection」，但数值上会明显破坏 `USHCN` 稳定性。

这一结果说明：

1. 不能把传播前节点特征幅值选择作为下一步主线。
2. `route_logit` 直接控制超图传播输入会重新放大坏轮。
3. 如果后续仍要保留「spike 负责传播选择」的论文叙事，必须改成更温和或更结构化的选择机制，例如 attention-level bias、残差式选择或 detach/stop-gradient 的解释性诊断，而不是直接缩放 observation message。
4. 当前工程主线仍应回到 `eventgateconst`，而不是升级到 `spikeselectprop_a1`。

补充实现状态：

- `spikeselectprop_a1` 的 n2h 输入缩放接线已撤回；
- 当前代码只保留 `selection_weight` 字段和 `QSHDiag` 中的 `selection_mean / selection_std` 输出；
- 这些字段仅作为后续结构诊断使用，不能视为传播选择结构已经进入主线。

## 2026-04-19：`spikeselectprop_res010` 与 `spikeselectprop_res005`

在 `spikeselectprop_a1` 失败后，重新分析失败原因：

- `selection_weight = sigmoid(route_logit)` 在初始 `route_logit = 0` 时等于 `0.5`；
- A1 直接把它乘到 n2h observation message 上，等价于初始化时把主传播消息砍半；
- 这破坏了 QSHNet 原先强调的 identity initialization。

因此下一步改成 identity-preserving 的 residual-style propagation selection：

```text
selection_factor = 1 + strength * (selection_weight - 0.5)
obs_selected = obs_base * selection_factor
```

这样当 `route_logit = 0` 时：

- `selection_weight = 0.5`
- `selection_factor = 1.0`
- 主传播路径初始化完全不变

### 1. `spikeselectprop_res010`

配置：

- `model_id = spikeselectprop_res010_itr10`
- `strength = 0.10`
- propagation factor 理论范围：`[0.95, 1.05]`

结果：

| 数据集 | 轮数 | MSE 均值 ± std | 单轮最大值 | `>0.18` | `>=0.20` | 结论 |
|------|------|----------------|------------|---------|----------|------|
| `USHCN` | 10 | `0.17535 ± 0.02540` | `0.24952` | 1 | 1 | 均值可接受，但仍有重坏轮 |

单轮 MSE：

- `0.160001`
- `0.166898`
- `0.160524`
- `0.169134`
- `0.167477`
- `0.178372`
- `0.166310`
- `0.249522`
- `0.160085`
- `0.175136`

### 2. `spikeselectprop_res005`

配置：

- `model_id = spikeselectprop_res005_itr10`
- `strength = 0.05`
- propagation factor 理论范围：`[0.975, 1.025]`

结果：

| 数据集 | 轮数 | MSE 均值 ± std | 单轮最大值 | `>0.18` | `>=0.20` | 结论 |
|------|------|----------------|------------|---------|----------|------|
| `USHCN` | 10 | `0.16988 ± 0.00937` | `0.19176` | 1 | 0 | 当前最好的 USHCN 尾部压制候选 |

补充 `HumanActivity itr=5` 对照：

| 数据集 | 轮数 | MSE 均值 ± std | 单轮最大值 | 结论 |
|------|------|----------------|------------|------|
| `HumanActivity` | 5 | `0.04175 ± 0.00018` | `0.04202` | 稳定，不伤简单数据 |

单轮 MSE：

- `0.191765`
- `0.166750`
- `0.159756`
- `0.168307`
- `0.166594`
- `0.178131`
- `0.164333`
- `0.169557`
- `0.158004`
- `0.175644`

诊断摘要：

- `L0_selection_factor_mean` 平均约 `0.99485`
- `L0_selection_factor_std` 平均约 `0.00393`
- `L0_quat_clip = 0.0`
- 没有 `>=0.20` 的重坏轮

### 结论

`spikeselectprop_res005` 是目前最值得保留的新候选。

它同时满足两个目标：

1. 数值上优于 `eventgateconst`
   - `eventgateconst`: `0.17587 ± 0.01571`, max `0.21558`
   - `spikeselectprop_res005`: `0.16988 ± 0.00937`, max `0.19176`

2. 叙事上比纯 `eventgateconst` 更好
   - spike 不再只是 event gate 的工程补丁；
   - 它以 identity-preserving residual bias 的形式参与超图传播；
   - hypergraph 仍是主干；
   - quaternion refinement 不变。

当前代码保留 `strength = 0.05` 的 residual-style propagation selection，作为下一步候选。

`HumanActivity` 对照进一步说明：

- `spikeselectprop_res005` 没有破坏简单数据集；
- 其 Human 表现与已知稳定候选 `eventdensvar_main` 的 `0.04174 ± 0.00019` 基本一致；
- 因此该结构已经通过本地两数据集初筛，下一步可以准备服务器覆盖 `P12 / MIMIC_III`。

## 2026-04-19：`memgradclip012` 失败试验

在 `routeconfvar` 失败后，进一步回到诊断里反复出现的核心风险信号：坏轮早期 `L0_membrane_w_grad` 偏高。

本轮只改一个训练动力学因素：

- 不改 forward；
- 不改 `route_logit` 输出；
- 不改 `retain`、`event`、`quaternion` 分支结构；
- 只给 `SpikeRouter.membrane_proj.weight/bias` 注册梯度 hook；
- 将 `membrane_proj` 梯度范数裁剪到 `0.12`。

运行版本：

- `model_id = memgradclip012_itr10`
- 日志：`storage/logs/qshdiag/ushcn_memgradclip012_itr10_20260419_094815.log`
- 解析 CSV：`storage/logs/qshdiag/ushcn_memgradclip012_itr10_20260419_094815_qshdiag.csv`

结果：

| 数据集 | 轮数 | MSE 均值 ± std | 单轮最大值 | 结论 |
|------|------|----------------|------------|------|
| `USHCN` | 10 | `0.17775 ± 0.02485` | `0.23575` | 均值尚可，但坏轮上界未解决 |

单轮 MSE：

- `iter0`: `0.163060`
- `iter1`: `0.164567`
- `iter2`: `0.165143`
- `iter3`: `0.163402`
- `iter4`: `0.163231`
- `iter5`: `0.235747`
- `iter6`: `0.164059`
- `iter7`: `0.217172`
- `iter8`: `0.170672`
- `iter9`: `0.170477`

早期 `QSHDiag` 对比显示：

- 坏轮的 `L0_route_logit_std` 仍然显著高于好轮；
- `epoch=2` 坏轮 `route_logit_std` 比好轮高约 `+0.3610`；
- `epoch=3` 坏轮 `route_logit_std` 比好轮高约 `+0.3014`；
- 坏轮 `L0_event_proj_w_norm` 仍然偏低；
- `membrane` 梯度虽然被裁剪，但 route dispersion 仍会快速拉大。

### 结论

`memgradclip012` 说明：坏轮不能通过直接裁剪 `membrane_proj` 梯度范数解决。

更准确的判断是：

1. `membrane_w_grad` 偏高是风险信号，但不是单独可控的充分因子。
2. 直接裁剪梯度没有阻断 route dispersion，反而仍出现 `0.23575` 级别坏轮。
3. 后续不应继续在 `membrane_proj` 上做简单梯度倍率或梯度裁剪。
4. 如果继续处理 route 动力学，需要设计更上层的机制，而不是单点 hook。

该试验代码已撤回，不进入主线。当前代码仍回到 `variable residual only + adaptive fused-cap` 主线。
