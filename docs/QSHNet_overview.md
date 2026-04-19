# QSH-Net 当前总览

> **最后更新：** 2026-04-19
> **用途：** 一页式入口文档。用于快速回答“当前模型是什么、当前保留哪版、当前效果怎样、下一步从哪里继续”。

## 1. 一句话结论

当前 QSH-Net 应表述为：

- `HyperIMTS` 超图主干
- `retain` 调制的连续脉冲路由
- 节点级 quaternion 残差增强
- 已被唤醒、已受控的 event 注入

而不应再表述为：

- `Spike + Event + Quaternion` 已经稳定充分协同

更准确的现实判断是：

- 真正长期活跃的是 `retain + quaternion + 已被唤醒的 event`
- `event` 已经不再是死分支
- 但 `event` 在高方差数据集上仍然存在明显尾部风险

## 2. 当前代码保留的是哪一版

当前工作区保留版本：

- `variable residual only + adaptive fused-cap`

它是在 `eventdensvar_main` 之后继续收缩得到的当前代码主线，核心差异是：

- temporal 路径保持不变
- event 注入的直接调制仍只保留在 variable 路径上
- fused context 阶段只约束 variable residual
- fused residual 上界会根据 route density 与 residual pressure 自适应收紧

对应代码位置：

- [QSHNet.py](/opt/Codes/PyOmniTS/models/QSHNet.py)
- [test_QSHNet.py](/opt/Codes/PyOmniTS/tests/models/test_QSHNet.py)

## 3. 当前模型的四个核心结构

### 3.1 超图主干

- 仍沿用 HyperIMTS 的 observation node / temporal hyperedge / variable hyperedge 三层表示
- 主消息传递骨架没有被推翻
- 当前所有有效改动都建立在“不破坏 HyperIMTS 主干”的前提上

### 3.2 脉冲部分

- 现在不是硬二值 fire 逻辑
- 本质是连续 route：
  - `retain_gate` 决定主路径保留强度
  - `event_gate` 决定事件路径注入强度
- 已确认 `retain_log_scale` 与 `membrane_proj.weight` 会持续学习

### 3.3 四元数部分

- 当前真正保留的是节点级 quaternion residual
- 通过 `quat_gate` 做条件化残差融合
- `quat_h2n` 与 `quat_gate` 已确认会持续学习
- 说明四元数分支不是挂名模块，而是当前真实活跃增强项之一

### 3.4 event 部分

- 旧 M1 中 `event` 曾经是死分支
- 当前版本已经把它唤醒
- 当前控制手段包括：
  - temporal / variable 双支路独立归一化
  - `event_scale` 温和上界
  - variable 路径的 density-aware 温和衰减

## 4. 当前版本层级怎么理解

### 4.1 统一主线母体

- `eventscalecap_main`
- `eventscalecap_itr10`

定位：

- 当前统一框架的默认对照版本
- 后续跨数据集归因实验默认相对它比较

### 4.2 当前工作区保留候选

- `variable residual only + adaptive fused-cap`

定位：

- 不是新的统一主线
- 但 USHCN `itr=5 = 0.16785 ± 0.00772` 已达到可接受状态
- 当前所有继续压 USHCN 坏轮的结构试验都应以它为直接起点

### 4.3 最保守稳定版本

- `retaincap_main`

定位：

- 如果只追求短期最稳工程结果，它仍然是最保守版本

## 5. 当前效果怎么看

### 5.1 统一主线母体

| 版本 | 数据集 | 轮数 | MSE 均值 ± std |
|------|--------|------|----------------|
| `eventscalecap_main` | HumanActivity | 3 | `0.0430 ± 0.0013` |
| `eventscalecap_main` | USHCN | 5 | `0.1663 ± 0.0027` |
| `eventscalecap_itr10` | USHCN | 10 | `0.1728 ± 0.0222` |

解读：

- 它仍是当前默认母体
- `itr=10` 下比 `eventnorm_itr10` 更稳
- 但 `USHCN` 长重复下仍有尾部坏轮

### 5.2 当前保留候选

| 版本 | 数据集 | 轮数 | MSE 均值 ± std |
|------|--------|------|----------------|
| `variable residual only + adaptive fused-cap` | USHCN | 5 | `0.16785 ± 0.00772` |
| `variable residual only + adaptive fused-cap` | USHCN | 10 | `0.19035 ± 0.03078` |
| `eventdensvar_main` | HumanActivity | 3 | `0.04181 ± 0.00011` |
| `eventdensvar_main` | USHCN | 5 | `0.1703 ± 0.0058` |
| `eventdensvar_main` | HumanActivity | 5 | `0.04174 ± 0.00019` |
| `eventdensvar_main` | USHCN | 10 | `0.1886 ± 0.0324` |
| `eventdensvar_main` | P12 | 5 | `0.30092 ± 0.00062` |
| `eventdensvar_main` | MIMIC_III | 5 | `0.39791 ± 0.01530` |

解读：

- 本地与服务器都确认 `HumanActivity` 改善被完整保住
- `P12` 在服务器上表现稳定
- `MIMIC_III` 均值可接受，但仍有单轮失稳
- `USHCN` 在服务器 `itr=10` 下坏轮仍明显，因此当前版本不能升级为统一主线
- 它仍然比同轮的其他 density 试验版本更可接受，因此保留为工作区候选仍然合理

### 5.3 已经明确否决的近邻版本

| 版本 | HumanActivity | USHCN | 结论 |
|------|---------------|-------|------|
| `eventrescap_main` | `0.04185 ± 0.00009` | `0.1834 ± 0.0273` | 全局 residual 比例硬约束失败 |
| `eventdenscap_main` | `0.04181 ± 0.00011` | `0.1891 ± 0.0347` | temporal + variable 全路径 density 抑制失败 |
| `routebound075` | 未跑 | `0.18132 ± 0.02602` | 单纯压 route logit 失败 |
| `eventprojgrad2` | 未跑 | `0.18962 ± 0.03279` | 单纯加速 event projection 学习失败 |
| `routeconfvar` | 未跑 | `0.19155 ± 0.03720` | route dispersion 感知 event 衰减失败 |
| `memgradclip012` | 未跑 | `0.17775 ± 0.02485` | 直接裁剪 membrane 梯度失败 |
| `routecenter` | 未跑 | `0.18433 ± 0.03669` | route 均值中心化失败 |

这两条结果说明：

- 不能继续做全局 `event` 收缩
- 如果继续沿 density 方向推进，只能看 variable 路径

## 6. 当前最重要的结构判断

1. 当前模型不是“超图没用，靠四元数和脉冲撑着”。
   - 超图主干仍然是骨架。

2. 当前模型也不是“三元素已经完全平衡协同”。
   - 真实情况更接近：超图主干 + retain 调制 + quaternion 残差 + 已被唤醒但仍需控尾的 event 注入。

3. `event` 已经从论文叙事模块变成真实训练模块。
   - 但它还没有完全稳定。

4. `EQHO` 目前仍然只是探索支线。
   - 它有研究价值，但当前不能替代主线。

## 7. 后续从哪里继续

如果下一步继续做架构优化，默认遵守下面三条：

1. 主线对照仍用 `eventscalecap_main / eventscalecap_itr10`
2. 当前代码实验从 `variable residual only + adaptive fused-cap` 接着做
3. 不再回到 `eventrescap_main` 或 `eventdenscap_main`
4. 不再默认继续做 route bound、event projection 梯度放大或 route dispersion 触发的 event 衰减
5. 不再默认继续做 `membrane_proj` 单点梯度裁剪或梯度倍率
6. 不再默认继续做 route logit 均值中心化

补充当前最直接的优化目标：

1. 不再追求进一步改善 `HumanActivity`
2. 优先压制 `USHCN` 的坏轮
3. 同时确保 `P12 / MIMIC_III` 不明显退化

## 8. 详细文档入口

- [QSHNet_naming_conventions.md](/opt/Codes/PyOmniTS/docs/QSHNet_naming_conventions.md)
- [QSHNet_results_summary.md](/opt/Codes/PyOmniTS/docs/QSHNet_results_summary.md)
- [QSHNet_server_validation.md](/opt/Codes/PyOmniTS/docs/QSHNet_server_validation.md)
- [QSHNet_architecture_status.md](/opt/Codes/PyOmniTS/docs/QSHNet_architecture_status.md)
- [hyperparameter_tuning_plan.md](/opt/Codes/PyOmniTS/docs/hyperparameter_tuning_plan.md)
- [QSHNet_evolution.md](/opt/Codes/PyOmniTS/docs/QSHNet_evolution.md)
- [QSHNet_current_design.md](/opt/Codes/PyOmniTS/docs/QSHNet_current_design.md)
  说明：这是 2026-04-15 的历史实现设计快照，不作为当前状态依据
