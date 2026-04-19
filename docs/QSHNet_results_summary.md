# QSH-Net 当前结果汇总

> **最后更新：** 2026-04-19
> **用途：** 单独汇总当前需要频繁引用的实验结果，避免在演化记录、调参计划和结构状态文档之间来回查找。

## 1. 使用原则

这份文档只汇总当前阶段最常用的结果表，不承担完整实验日志职责。

使用时默认遵守：

- 当前工作区主线：`variable residual only + adaptive fused-cap`
- 当前可接受本地结果：`USHCN itr=5 = 0.16785 ± 0.00772`
- 已明确失败的近邻版本：`quat ceiling / retain ceiling / routebound075 / eventprojgrad2 / routeconfvar / memgradclip012 / routecenter`

更完整的上下文说明见：

- [QSHNet_naming_conventions.md](/opt/Codes/PyOmniTS/docs/QSHNet_naming_conventions.md)
- [QSHNet_overview.md](/opt/Codes/PyOmniTS/docs/QSHNet_overview.md)
- [QSHNet_server_validation.md](/opt/Codes/PyOmniTS/docs/QSHNet_server_validation.md)
- [QSHNet_architecture_status.md](/opt/Codes/PyOmniTS/docs/QSHNet_architecture_status.md)
- [hyperparameter_tuning_plan.md](/opt/Codes/PyOmniTS/docs/hyperparameter_tuning_plan.md)
- [QSHNet_evolution.md](/opt/Codes/PyOmniTS/docs/QSHNet_evolution.md)

## 2. 当前最常引用的版本

| 版本 | 定位 | 说明 |
|------|------|------|
| `retaincap_main` | 最保守稳定版本 | 如果只追求短期最稳工程结果，优先参考它 |
| `eventscalecap_main` | 历史统一主线母体 | 仍可作为早期三元素框架对照 |
| `eventscalecap_itr10` | 历史长重复母体 | 仍可作为长重复对照 |
| `eventdensvar_main` | 历史 density-aware 候选 | 服务器验证后不能升级为统一主线 |
| `coupledctxadapt_main` | 当前代码主线语义 | `variable residual only + adaptive fused-cap`，当前工作区默认结构 |

## 3. 主线与保留候选结果

### 3.1 HumanActivity

| 版本 | 轮数 | MSE 均值 ± std | 结论 |
|------|------|----------------|------|
| `eventnorm_main` | 3 | `0.0430 ± 0.0013` | 三元素统一框架的早期有效版本 |
| `eventscalecap_main` | 3 | `0.0430 ± 0.0013` | 当前统一主线母体 |
| `eventdensvar_main` | 3 | `0.04181 ± 0.00011` | 当前保留候选，在 HumanActivity 上继续改善 |
| `eventdensvar_main` | 5 | `0.04174 ± 0.00019` | 服务器验证确认改善可复现，且稳定性保持良好 |

### 3.2 USHCN

| 版本 | 轮数 | MSE 均值 ± std | 结论 |
|------|------|----------------|------|
| `retaincap_main` | 5 | `0.1673 ± 0.0033` | 最保守稳定版本 |
| `eventnorm_main` | 5 | `0.1653 ± 0.0011` | 短重复下数值最佳的三元素版本之一 |
| `eventscalecap_main` | 5 | `0.1663 ± 0.0027` | 当前统一主线母体 |
| `eventdensvar_main` | 5 | `0.1703 ± 0.0058` | 略弱于主线，但已达到可接受保留水平 |
| `eventnorm_itr10` | 10 | `0.1829 ± 0.0279` | 长重复下尾部风险明显 |
| `eventscalecap_itr10` | 10 | `0.1728 ± 0.0222` | 当前长重复主线母体，优于 `eventnorm_itr10` |
| `eventdensvar_main` | 10 | `0.1886 ± 0.0324` | 服务器验证显示坏轮仍明显，不能升级为统一主线 |
| `variable residual only + adaptive fused-cap` | 5 | `0.16785 ± 0.00772` | 用户已接受，当前代码回退到此主线 |
| `variable residual only + adaptive fused-cap` | 10 | `0.19035 ± 0.03078` | 诊断运行显示后段坏轮仍存在 |
| `routebound075` | 10 | `0.18132 ± 0.02602` | 压 route logit 不足以解决坏轮，已撤回 |
| `eventprojgrad2` | 10 | `0.18962 ± 0.03279` | 加速 event projection 学习失败，已撤回 |
| `routeconfvar` | 10 | `0.19155 ± 0.03720` | route dispersion 感知 event 衰减失败，已撤回 |
| `memgradclip012` | 10 | `0.17775 ± 0.02485` | 均值尚可，但坏轮上界仍到 `0.23575`，已撤回 |
| `routecenter` | 10 | `0.18433 ± 0.03669` | route 均值中心化失败，坏轮上界到 `0.27430`，已撤回 |
| `eventgateconst` | 10 | `0.17587 ± 0.01571` | 当前最值得保留的尾部压制候选，显著优于多数失败近邻版本 |
| `eventgateconst_quat020` | 10 | `0.17550 ± 0.01473` | 数值略好于 `eventgateconst`，但 `quat_clip=0.0`，说明不是根因级改动，已撤回 |
| `eventgateconst_meminit001` | 10 | `0.21388 ± 0.08635` | 明显失败，出现 `0.46190` 级别坏轮，已撤回 |
| `eventgateconst_routedetach` | 10 | `0.18122 ± 0.02408` | 切断 route density 反向梯度后退化，已撤回 |

### 3.3 P12

| 版本 | 轮数 | MSE 均值 ± std | 结论 |
|------|------|----------------|------|
| `eventdensvar_main` | 5 | `0.30092 ± 0.00062` | 服务器验证稳定，通过跨数据集可用性检查 |

### 3.4 MIMIC_III

| 版本 | 轮数 | MSE 均值 ± std | 结论 |
|------|------|----------------|------|
| `eventdensvar_main` | 5 | `0.39791 ± 0.01530` | 均值可接受，但仍有单轮失稳，稳定性未完全收口 |

## 4. 本轮 `event density` 试验链结果

### 4.1 HumanActivity

| 版本 | 轮数 | MSE 均值 ± std | 结论 |
|------|------|----------------|------|
| `eventrescap_main` | 3 | `0.04185 ± 0.00009` | HumanActivity 改善，但不足以证明方向成立 |
| `eventdenscap_main` | 3 | `0.04181 ± 0.00011` | HumanActivity 继续保持改善 |
| `eventdensvar_main` | 3 | `0.04181 ± 0.00011` | 最终保留候选 |

### 4.2 USHCN

| 版本 | 轮数 | MSE 均值 ± std | 结论 |
|------|------|----------------|------|
| `eventrescap_main` | 5 | `0.1834 ± 0.0273` | 全局 residual 比例硬约束失败 |
| `eventdenscap_main` | 5 | `0.1891 ± 0.0347` | temporal + variable 全路径 density 抑制失败 |
| `eventdensvar_main` | 5 | `0.1703 ± 0.0058` | 只保留 variable 路径 density 控制后，结果收敛到可接受水平 |

## 5. EQHO 支线结果

### 5.1 HumanActivity

| 版本 | 轮数 | MSE 均值 ± std | 结论 |
|------|------|----------------|------|
| `QSHNet_EQHO_A3` | 3 | `0.0418 ± 0.0002` | 可训练，可运行 |
| `QSHNet_EQHO_A25` | 3 | `0.0418 ± 0.0002` | 受限动态化不伤简单数据 |
| `QSHNet_EQHO_A26` | 3 | `0.0418 ± 0.0003` | gain hard cap 不伤简单数据 |
| `QSHNet_EQHO_S1` | 3 | `0.0418 ± 0.0003` | 输出归一化不伤简单数据 |
| `QSHNet_EQHO_S2` | 3 | `0.0422 ± 0.0002` | 已开始伤简单数据 |
| `QSHNet_EQHO_S3` | 3 | `0.0418 ± 0.0004` | residual-style summarizer 回到安全线附近 |

### 5.2 USHCN

| 版本 | 轮数 | MSE 均值 ± std | 结论 |
|------|------|----------------|------|
| `QSHNet_EQHO_A3` | 5 | `0.1979 ± 0.0310` | full dynamic mix_coef 明显失稳 |
| `QSHNet_EQHO_A25` | 5 | `0.2062 ± 0.0264` | 压住 mix_coef 后仍明显退化 |
| `QSHNet_EQHO_A26` | 5 | `0.1846 ± 0.0297` | 优于 A25/A3，但仍明显弱于主线 |
| `QSHNet_EQHO_S1` | 5 | `0.2072 ± 0.0461` | 只压输出尺度无效 |
| `QSHNet_EQHO_S2` | 5 | `0.2218 ± 0.0336` | 分路静态聚合失败 |
| `QSHNet_EQHO_S3` | 5 | `0.2173 ± 0.0340` | residual-style summarizer 仍失败 |

## 6. 当前推荐引用口径

如果后续需要快速引用“当前模型效果”，默认优先使用下面四条：

1. `eventscalecap_main`
   - HumanActivity `0.0430 ± 0.0013`
   - USHCN `0.1663 ± 0.0027`

2. `eventscalecap_itr10`
   - USHCN `0.1728 ± 0.0222`

3. `variable residual only + adaptive fused-cap`
   - USHCN 本地 `itr=5`: `0.16785 ± 0.00772`
   - USHCN 本地诊断 `itr=10`: `0.19035 ± 0.03078`
   - 当前代码主线，继续优化时以它为准

4. `eventdensvar_main`
   - HumanActivity 本地 `0.04181 ± 0.00011`
   - HumanActivity 服务器 `0.04174 ± 0.00019`
   - USHCN 本地 `0.1703 ± 0.0058`
   - USHCN 服务器 `0.1886 ± 0.0324`
   - P12 服务器 `0.30092 ± 0.00062`
   - MIMIC_III 服务器 `0.39791 ± 0.01530`

5. `retaincap_main`
   - USHCN `0.1673 ± 0.0033`

## 7. 当前结果层级结论

1. 如果要讲“历史统一主线”，引用 `eventscalecap_main / eventscalecap_itr10`。
2. 如果要讲“当前代码工作区主线”，引用 `variable residual only + adaptive fused-cap`。
3. 如果要讲“最稳保守版本”，引用 `retaincap_main`。
4. 如果要讲“为什么不继续走全局 density / residual 收缩”，引用 `eventrescap_main / eventdenscap_main`。
5. 如果要讲“为什么 EQHO 目前不能替代主线”，引用 `QSHNet_EQHO_A26` 及其后续 `S1/S2/S3`。
6. 如果要讲“这次服务器全数据集验证的正式结论”，引用：
   - `HumanActivity` 和 `P12` 通过
   - `MIMIC_III` 基本可接受但仍有坏轮
   - `USHCN` 未通过，因此 `eventdensvar_main` 不能升级为新的统一主线
7. 如果要讲“最近为什么不继续压单一分支”，引用：
   - `routebound075`: `0.18132 ± 0.02602`, max `0.25184`
   - `eventprojgrad2`: `0.18962 ± 0.03279`, max `0.26653`
   - `routeconfvar`: `0.19155 ± 0.03720`, max `0.26306`
   - `memgradclip012`: `0.17775 ± 0.02485`, max `0.23575`
   - `routecenter`: `0.18433 ± 0.03669`, max `0.27430`
   - `eventgateconst_quat020`: `0.17550 ± 0.01473`, max `0.21027`
   - `eventgateconst_meminit001`: `0.21388 ± 0.08635`, max `0.46190`
   - `eventgateconst_routedetach`: `0.18122 ± 0.02408`, max `0.23287`
   - `spikeselectprop_a1`: `0.19942 ± 0.02713`, max `0.26434`
   - `spikeselectprop_res010`: `0.17535 ± 0.02540`, max `0.24952`
   - `spikeselectprop_res005`: `0.16988 ± 0.00937`, max `0.19176`

## 8. 当前最该引用的最新 USHCN 判断

如果只想快速说明“最近这轮结构试验得到什么结论”，优先引用下面 3 条：

1. `eventgateconst`
- `USHCN itr=10 = 0.17587 ± 0.01571`, max `0.21558`
- 当前唯一明确值得保留的尾部压制候选

2. `eventgateconst_meminit001`
- `USHCN itr=10 = 0.21388 ± 0.08635`, max `0.46190`
- 明确否定“主动提高 route 初始分散度”这一方向

3. `eventgateconst_routedetach`
- `USHCN itr=10 = 0.18122 ± 0.02408`, max `0.23287`
- 明确否定“切断 route density 到稳定化分支的训练反馈”这一方向

4. `spikeselectprop_a1`
- `USHCN itr=10 = 0.19942 ± 0.02713`, max `0.26434`
- 明确否定“直接用 `sigmoid(route_logit)` 缩放 n2h observation message”这一传播选择实现
- 说明论文友好的 spike 叙事不能靠简单输入幅值选择落地；后续若继续做，应转向更温和的结构选择机制

5. `spikeselectprop_res005`
- `USHCN itr=10 = 0.16988 ± 0.00937`, max `0.19176`
- `HumanActivity itr=5 = 0.04175 ± 0.00018`, max `0.04202`
- 当前最值得引用的新候选
- 相比 `eventgateconst`，均值、方差和最差轮都更好
- 机制上保留 identity initialization，并让 spike 以 residual propagation bias 形式参与超图传播
