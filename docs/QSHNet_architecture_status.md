# QSH-Net 当前模型架构状态总览

> **最后更新：** 2026-04-19  
> **用途：** 作为后续继续做结构试验时的唯一基准文档。优先级高于口头结论与零散实验记录。

## 1. 当前真实动力学

当前 QSH-Net 的真实训练动力学，不应再表述为：

- `Spike + Event + Quaternion` 已经稳定协同

更准确的表述是：

- `HyperIMTS + retain 调制 + quaternion 残差 + 已被唤醒且受控的 event 注入`

其中：

- 超图主干仍然承担主要结构建模
- `retain` 与 quaternion 仍然是持续活跃的主增强路径
- `event` 已经不再是死分支
- 但 `event` 仍然具有明显的数据集依赖尾部风险

## 2. 已经确认的基础事实

### 2.1 旧 M1 的真实问题

旧 M1 已通过本地参数诊断确认：

- `event_proj.weight norm = 0.0`
- `event_residual_scale = 0.0`
- 在 `USHCN` 与 `HumanActivity` 上都成立

这说明：

- 旧 M1 的 `event residual` 分支最初是死分支
- 当时真正持续学习的是：
  - `retain_log_scale`
  - `membrane_proj.weight`
  - `quat_gate`
  - `quat_h2n`

### 2.2 USHCN 的核心难点

`USHCN` 的坏轮更接近：

- 真实活跃分支在高方差数据上被放大

而不是：

- 单纯的 `event gate` 没调好
- 常规学习率 / batch size 问题

## 3. 当前代码里真正保留的结构基线

当前应当视为“活跃基线”的结构包括：

1. `retain` 温和约束
- `retain_strength_max = 0.1`

2. quaternion 残差保留
- 节点级 quaternion gate
- quaternion residual ratio bound

3. `event` 支路已被激活
- `SpikeRouter.event_log_scale = -4.5`
- `event_residual_scale` 采用 `sigmoid` 语义，而不是旧的近零残差语义

4. `event` 双支路独立归一化
- `temporal_event_norm`
- `variable_event_norm`

5. `event` 总注入量温和上界
- `event_scale = clamp(sigmoid(event_residual_scale), max=0.12)`

6. `variable` 路径允许基于 route density 做温和衰减
- `event_density_baseline = 0.5`
- `temporal_event_density_penalty_max = 0.0`
- `variable_event_density_penalty_max = 0.5`

更准确地说，历史上 `eventdensvar_main` 是 density-aware 方向的可接受候选；当前代码已经继续推进到：

- `variable residual only + adaptive fused-cap`

当前代码语义：

- temporal event 注入路径保持不变
- variable event 注入仍保留 density-aware 温和衰减
- fused context 阶段只约束 variable residual

### 3.1 2026-04-19 的代码级补充状态

在最近一轮 `USHCN` 尾部压制试验之后，当前代码工作区的真实状态应补充为：

1. 当前唯一保留的新候选是 `eventgateconst`
- `event_gate` 已与 `route_logit` 解耦
- 当前实现为：
  - `event_gate = 0.5 * exp(event_log_scale)`

2. 下面这些改动都已经证伪并回退
- `quat_residual_ratio_max = 0.20`
- `membrane_proj` 小随机初始化
- `route_density` 反向梯度切断

3. 因此当前代码并不是新的综合稳定化版本，而是：
- `eventgateconst + 原有 variable residual only + adaptive fused-cap`

## 4. 2026-04-19 之后新增的结构试验结论

### 4.1 当前最值得保留的候选：`eventgateconst`

`USHCN itr=10`：

- 均值 `0.17587`
- std `0.01571`
- max `0.21558`

它的意义不是“问题已经解决”，而是：

- 这是当前唯一一个在不引入更大副作用的前提下，实质压缩了坏轮尾部的单因素结构改动

### 4.2 已证伪的近邻方向

1. `eventgateconst_quat020`
- 均值 `0.17550`
- std `0.01473`
- max `0.21027`
- 但 `quat_clip = 0.0`
- 说明 cap 没真正介入，已回退

2. `eventgateconst_meminit001`
- 均值 `0.21388`
- std `0.08635`
- max `0.46190`
- 明显失败，已回退

3. `eventgateconst_routedetach`
- 均值 `0.18122`
- std `0.02408`
- max `0.23287`
- 明显失败，已回退

## 5. 对当前架构的收敛判断

截至当前，最准确的架构判断应为：

1. 当前模型仍然不是稳定成熟的“三元素完全协同”结构。
2. `event` 分支已经比旧 M1 更活跃，但仍不是稳定可靠的增益主力。
3. `retain + quaternion + HyperIMTS backbone` 仍然构成主导动力学。
4. `eventgateconst` 说明：
   - route 不适合继续直接控制 event 幅值
   - 但 route 也不能被简单地从稳定化反馈中切断
5. 后续若继续做结构试验，不应再优先碰：
   - `membrane` 初始化
   - route density detach
   - 纯 quaternion residual cap
- fused residual 上界根据 fused route density 与 residual pressure 自适应收紧
- 已失败并撤回的 `routebound075`、`eventprojgrad2`、`routeconfvar` 不属于当前代码
- 已失败并撤回的 `memgradclip012` 也不属于当前代码
- 已失败并撤回的 `routecenter` 也不属于当前代码

## 4. 当前主线候选版本

### 4.1 最稳的保守版本

- `retaincap_main`

适用定位：

- 如果目标只是追求短期最稳的工程结果

### 4.2 当前统一实验母体

- `eventnorm + mild event_scale cap`
- 对应实验标识：
  - `eventscalecap_main`
  - `eventscalecap_itr10`

当前最准确的判断：

- 它优于单纯的 `eventnorm_itr10`
- 它是目前冻结后的统一实验母体
- 后续所有跨数据集归因实验默认都相对 `eventscalecap_main` / `eventscalecap_itr10` 比较

### 4.3 当前可接受保留候选

- `variable residual only + adaptive fused-cap`

适用定位：

- 如果目标是继续压制 `USHCN` 坏轮，同时不回到全局收缩或单分支 ceiling

当前判断：

- 它不是新的统一母体
- 但本地 `USHCN itr=5 = 0.16785 ± 0.00772` 已被接受
- 诊断 `USHCN itr=10 = 0.19035 ± 0.03078` 说明坏轮仍未完全解决
- 后续若继续结构优化，应默认以它为直接起点

## 5. 从上次文档更新后到现在的完整试验链

### 5.1 成功保留的方向

1. `eventnorm_main`
- 说明 `event` 活化后，独立归一化是关键控制手段

2. `eventscalecap_main`
- 说明限制 `event` 总注入量是安全方向

3. `eventscalecap_itr10`
- 说明温和上界对长重复稳定性有真实帮助

4. `eventdensvar_main`
- 说明“只在 variable 路径上做 density-aware event_scale 衰减”是这一轮 density 试验链里唯一值得保留的版本

5. `variable residual only + adaptive fused-cap`
- 说明继续把稳定化收窄到 fused variable residual 更安全
- 当前本地 `USHCN itr=5 = 0.16785 ± 0.00772`，已达到可接受线

### 5.2 已明确否决的方向

1. `eventgain_main`
- 额外主干条件化 gain
- 结论：失败

2. `event_temporal_only`
- 只保留 temporal 注入
- 结论：失败

3. `eventnorm_clip`
- 直接裁剪归一化后的 `event_delta`
- 结论：失败

4. `eventrescap_main`
- 直接把 `event residual` 总量相对主状态做比例硬约束
- 结论：失败

5. `eventdenscap_main`
- 对 temporal / variable 两条路径同时做 density-aware 抑制
- 结论：失败

6. `routebound075`
- 单纯压 `route_logit` 分布
- `USHCN itr=10 = 0.18132 ± 0.02602`, max `0.25184`
- 结论：失败，已撤回

7. `eventprojgrad2`
- 单纯把 `event_proj` 梯度放大 `2x`
- `USHCN itr=10 = 0.18962 ± 0.03279`, max `0.26653`
- 结论：失败，已撤回

8. `routeconfvar`
- route dispersion 感知的 variable event residual 衰减
- `USHCN itr=10 = 0.19155 ± 0.03720`, max `0.26306`
- 结论：失败，已撤回

9. `memgradclip012`
- 直接裁剪 `membrane_proj` 梯度范数到 `0.12`
- `USHCN itr=10 = 0.17775 ± 0.02485`, max `0.23575`
- 结论：坏轮上界未解决，已撤回

10. `routecenter`
- 对 `route_logit` 做样本内均值中心化
- `USHCN itr=10 = 0.18433 ± 0.03669`, max `0.27430`
- 结论：坏轮更差，已撤回

这些失败方向共同说明：

- 不能靠增加额外控制器来“驯服 event”
- 不能靠简单删支路来“缩小传播范围”
- 不能靠粗暴裁剪 `event_delta` 形状来消除坏轮

真正更有效的方向是：

- 保留双支路表达
- 保留独立归一化
- 对总注入量做平滑而轻量的控制
- 若引入 density-aware 控制，只在更可能放大噪声的 variable 路径做

## 6. 当前数值结论

### 6.1 `USHCN itr=5`

- `eventnorm_main`: `0.1653 ± 0.0011`
- `eventscalecap_main`: `0.1663 ± 0.0027`

解读：

- `event_scale cap` 在短重复下没有显著超过 `eventnorm_main`
- 但它是安全改动，没有破坏当前框架

### 6.2 `USHCN itr=10`

- `eventnorm_itr10`: `0.1829 ± 0.0279`
- `eventscalecap_itr10`: `0.1728 ± 0.0222`

解读：

- `event_scale cap` 对长重复稳定性有真实改善
- 但最差轮仍然达到 `0.2355`
- 因此它属于“明显改善”，不是“最终解决”

### 6.3 冻结对照表

| 基线 | 数据集 | 轮数 | MSE 均值 ± std | 备注 |
|------|--------|------|----------------|------|
| `eventscalecap_main` | HumanActivity | 3 | `0.0430 ± 0.0013` | 当前统一实验母体的一部分 |
| `eventscalecap_main` | USHCN | 5 | `0.1663 ± 0.0027` | 当前统一实验母体的一部分 |
| `eventscalecap_itr10` | USHCN | 10 | `0.1728 ± 0.0222` | 长重复默认对照 |
| `eventrescap_main` | HumanActivity | 3 | `0.04185 ± 0.00009` | HumanActivity 改善，但不具备主线价值 |
| `eventrescap_main` | USHCN | 5 | `0.1834 ± 0.0273` | 均值和方差都明显退化，已否决 |
| `eventdenscap_main` | HumanActivity | 3 | `0.04181 ± 0.00011` | HumanActivity 继续保持改善 |
| `eventdenscap_main` | USHCN | 5 | `0.1891 ± 0.0347` | 全路径 density 抑制最差，已否决 |
| `eventdensvar_main` | HumanActivity | 3 | `0.04181 ± 0.00011` | 保持 HumanActivity 改善 |
| `eventdensvar_main` | USHCN | 5 | `0.1703 ± 0.0058` | 略弱于主线，但方差明显收敛，当前可接受保留候选 |
| `eventdensvar_main` | HumanActivity | 5 | `0.04174 ± 0.00019` | 服务器验证确认改善稳定可复现 |
| `eventdensvar_main` | USHCN | 10 | `0.1886 ± 0.0324` | 服务器验证显示坏轮仍明显，不能升级为统一主线 |
| `eventdensvar_main` | P12 | 5 | `0.30092 ± 0.00062` | 服务器验证稳定，通过跨数据集可用性检查 |
| `eventdensvar_main` | MIMIC_III | 5 | `0.39791 ± 0.01530` | 均值可接受，但仍有单轮失稳 |
| `variable residual only + adaptive fused-cap` | USHCN | 5 | `0.16785 ± 0.00772` | 当前代码主线，用户已接受 |
| `variable residual only + adaptive fused-cap` | USHCN | 10 | `0.19035 ± 0.03078` | 诊断运行显示后段坏轮仍存在 |
| `routebound075` | USHCN | 10 | `0.18132 ± 0.02602` | 单纯压 route logit 失败，已撤回 |
| `eventprojgrad2` | USHCN | 10 | `0.18962 ± 0.03279` | 加速 event projection 学习失败，已撤回 |
| `routeconfvar` | USHCN | 10 | `0.19155 ± 0.03720` | route dispersion 衰减失败，已撤回 |
| `memgradclip012` | USHCN | 10 | `0.17775 ± 0.02485` | membrane 梯度裁剪失败，已撤回 |
| `routecenter` | USHCN | 10 | `0.18433 ± 0.03669` | route 均值中心化失败，已撤回 |
| `spikeselectprop_a1` | USHCN | 10 | `0.19942 ± 0.02713` | 直接用 `selection_weight` 缩放 n2h observation message 明显退化，不进入主线 |
| `spikeselectprop_res010` | USHCN | 10 | `0.17535 ± 0.02540` | identity-preserving residual selection，均值可接受但仍有 `0.24952` 重坏轮 |
| `spikeselectprop_res005` | USHCN | 10 | `0.16988 ± 0.00937` | 当前最好的 USHCN 尾部压制候选，且叙事优于纯 `eventgateconst` |
| `spikeselectprop_res005` | HumanActivity | 5 | `0.04175 ± 0.00018` | 稳定，不伤简单数据 |
| `QSHNet_EQHO_A3` | HumanActivity | 3 | `0.0418 ± 0.0002` | `EQHO` 在简单数据上可训练、可运行 |
| `QSHNet_EQHO_A3` | USHCN | 5 | `0.1979 ± 0.0310` | `full dynamic mix_coef` 明显失稳，不能入主线 |
| `QSHNet_EQHO_A25` | HumanActivity | 3 | `0.0418 ± 0.0002` | 受限动态化保持可训练，不破坏简单数据表现 |
| `QSHNet_EQHO_A25` | USHCN | 5 | `0.2062 ± 0.0264` | `mix_coef` 已被压回安全区，但 `mix_gain` 仍导致坏轮 |
| `QSHNet_EQHO_A26` | HumanActivity | 3 | `0.0418 ± 0.0003` | `gain hard cap` 不伤简单数据，仍可训练 |
| `QSHNet_EQHO_A26` | USHCN | 5 | `0.1846 ± 0.0297` | 优于 `A25/A3`，但仍明显弱于主线 |
| `QSHNet_EQHO_S1` | HumanActivity | 3 | `0.0418 ± 0.0003` | `event_summary` 输出归一化不伤简单数据 |
| `QSHNet_EQHO_S1` | USHCN | 5 | `0.2072 ± 0.0461` | 虽压低 `summary` 统计量，但结果更差，方向失败 |
| `QSHNet_EQHO_S2` | HumanActivity | 3 | `0.0422 ± 0.0002` | 分路静态聚合已伤简单数据，不可保留 |
| `QSHNet_EQHO_S2` | USHCN | 5 | `0.2218 ± 0.0336` | 明显更差，且破坏 real-dominant 结构 |
| `QSHNet_EQHO_S3` | HumanActivity | 3 | `0.0418 ± 0.0004` | residual-style summarizer 回到安全线附近 |
| `QSHNet_EQHO_S3` | USHCN | 5 | `0.2173 ± 0.0340` | 仍明显退化，不能替代 `A2.6` |

## 7. 当前统一表述

后续继续讨论模型时，应统一使用下面这句作为基准：

> QSH-Net 当前已经形成一个由超图主干、四元数增强与事件感知注入组成的统一框架，其中事件支路已从静默模块变成真实可训练、可控，并可通过轻量总量约束改善长重复稳定性的结构组成部分；但在 `USHCN` 这类高方差数据上，长重复下仍然存在尾部坏轮，因此当前默认统一主线仍应以 `eventscalecap_main / eventscalecap_itr10` 为准，而 `eventdensvar_main` 仅作为保留候选，不应视为最终稳定定版。

补充说明：

- `event density` 试验链已经给出一条明确约束：不要对 temporal 路径做 density-aware 抑制；
- `eventrescap_main` 与 `eventdenscap_main` 都证明“更强的全局 event 收缩”会明显伤到 `USHCN`；
- `eventdensvar_main` 则进一步说明：把 density-aware 控制收窄到 variable 路径后，虽然均值仍略弱于 `eventscalecap_main`，但已经把方差压到 `0.0058` 量级，可视为可接受保留候选；
- 但服务器四数据集验证进一步说明：这种“可接受”主要成立于本地筛选阶段；一旦进入 `USHCN itr=10` 的正式验证，坏轮仍会重新出现，因此它只能作为保留候选，不能当作统一主线；
- 同时服务器结果也证明：该结构并非全面失败，因为 `HumanActivity` 和 `P12` 稳定通过，`MIMIC_III` 也保持在基本可接受区间；
- `EQHO` 探索已经证明 hyperedge-level event-conditioned quaternion refinement 是可训练的；
- 但 `A3` 的完整动态 `mix_coef` 会在 `USHCN` 上快速偏离 real-dominant 安全区，导致 `gain`、`summary` 与 `residual` 同时抬升；
- `A2.5` 进一步证明：即便用 safety floor 把 `mix_coef` 压回 real-dominant 安全区，`USHCN` 上仍会因为 `mix_gain` 接近饱和而出现显著坏轮；
- `A2.6` 再进一步证明：把 `mix_gain` 上界压到 `0.08` 后，`USHCN` 确实改善，但 `summary` 与 `residual` 仍然偏高，说明更深层的放大源头仍在 `event_summary` 本身；
- `S1` 继续证明：即使把 `event_summary` 输出端直接归一化，`USHCN` 结果仍会退化到 `0.2072 ± 0.0461`，说明“只压平输出尺度”不是有效解；
- `S2` 继续证明：直接重写为分路静态加权聚合会破坏 real-dominant 安全结构，本身就是坏方向；
- `S3` 继续证明：即使把 summarizer 改成主干主导的 bounded residual 结构，`USHCN` 仍退化到 `0.2173 ± 0.0340`；
- 因此 `EQHO` 当前只能作为探索支线，不能替代 `eventscalecap_main / eventscalecap_itr10` 的主线地位。

## 8. 后续试验约束

后续若继续做结构试验，默认遵守以下规则：

1. 每次只改一个核心因素
2. 优先使用：
- `USHCN`
- `HumanActivity`

3. 结论标准：
- `HumanActivity` 可用 `itr=3`
- `USHCN` 至少 `itr=5`
- 更稳妥的最终判断需要 `itr=10`

4. 不再优先做：
- 常规超参数扫描
- 无差别增强 `event` 强度
- 回到已经明确否决的 `eventgain` / `temporal_only` / `eventnorm_clip`

5. 当前最优先的结构目标：
- 只针对 `USHCN` 坏轮做抑制
- 不以牺牲 `HumanActivity / P12` 的已确认收益为代价

6. 后续所有跨数据集归因实验默认相对 `eventscalecap_main` / `eventscalecap_itr10` 比较。

7. 不再回到已经明确否决的 `eventrescap_main` / `eventdenscap_main`。

8. `EQHO` 当前仍是冻结支线：
- 可以作为研究储备
- 但不应占用当前主线优化资源
- 除非后续重新提出更上层的新结构假设，否则不再优先继续沿 `A3 / A2.5 / A2.6 / S1 / S2 / S3` 细调

9. `spikeselectprop_a1` 已证明「传播选择」不能简单实现为输入特征幅值缩放：
- 该版本虽然更贴近「spike-driven propagation selection」论文叙事；
- 但 `USHCN itr=10` 退化到 `0.19942 ± 0.02713`，max `0.26434`；
- 因此不能把它升级为当前默认架构；
- 若后续继续做论文友好的 spike 结构，应避免直接缩放 observation message，改为更温和的结构选择或 attention-level 机制。

10. `spikeselectprop_res005` 是当前新的结构候选：
- 它把传播选择改成 `1 + 0.05 * (selection_weight - 0.5)`；
- `route_logit = 0` 时保持 identity，不破坏 HyperIMTS 主干初始化；
- `USHCN itr=10 = 0.16988 ± 0.00937`，max `0.19176`；
- `HumanActivity itr=5 = 0.04175 ± 0.00018`，与已知稳定候选持平；
- 相比 `eventgateconst` 的 `0.17587 ± 0.01571`，均值、方差和最差轮都更好；
- 已通过本地 `USHCN + HumanActivity` 初筛，下一步可以送服务器覆盖 `P12 / MIMIC_III`。

## 9. 相关文档

- [当前总览](./QSHNet_overview.md)
- [标准命名与引用规范](./QSHNet_naming_conventions.md)
- [当前结果汇总](./QSHNet_results_summary.md)
- [服务器验证说明](./QSHNet_server_validation.md)
- [QSHNet 演化记录](./QSHNet_evolution.md)
- [超参数与诊断计划](./hyperparameter_tuning_plan.md)
- [论文叙事草稿](./QSHNet_paper_narrative.md)
