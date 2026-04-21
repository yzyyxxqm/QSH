# QSH-Net 当前结果汇总

> **最后更新：** 2026-04-21
> **用途：** 汇总当前阶段最常被引用的实验结果，避免在演化记录、调参计划与结构状态文档之间反复查找。

## 1. 当前最常引用的基线

### 1.1 HyperIMTS 论文/项目参考结果

| 数据集 | 轮数 | MSE 均值 ± std | MAE 均值 ± std |
|------|------|----------------|----------------|
| `USHCN` | 5 | `0.1738 ± 0.0078` | `0.2773 ± 0.0064` |
| `HumanActivity` | 5 | `0.0421 ± 0.0021` | `0.1199 ± 0.0059` |
| `P12` | 5 | `0.2996 ± 0.0003` | `0.3598 ± 0.0002` |
| `MIMIC_III` | 5 | `0.4259 ± 0.0021` | `0.3800 ± 0.0009` |
| `MIMIC_IV` | 5 | `0.21740 ± 0.00090` | `0.28370 ± 0.00070` |

### 1.2 当前 `res005` 论文候选参考结果

| 数据集 | 轮数 | MSE 均值 ± std | MAE 均值 ± std |
|------|------|----------------|----------------|
| `USHCN` | 5 主表 / 10 压测 | `0.16750 ± 0.00357` | `0.27023 ± 0.00618` |
| `HumanActivity` | 5 | `0.04172 ± 0.00018` | `0.11571 ± 0.00107` |
| `P12` | 5 | `0.30087 ± 0.00084` | `0.36120 ± 0.00087` |
| `MIMIC_III` | 5 | `0.39396 ± 0.00308` | `0.37381 ± 0.00226` |
| `MIMIC_IV` | 5 | `0.21549 ± 0.00188` | `0.27911 ± 0.00188` |

## 2. 当前工作区最近主线结果

### 2.1 `HumanActivity`

| 版本 | 轮数 | MSE 均值 ± std | MAE 均值 ± std | 结论 |
|------|------|----------------|----------------|------|
| `qsh_rescorrnorm100_human_d128l3_e300_local` | 5 | `0.041469 ± 0.000171` | `0.114428 ± 0.001061` | 当前本地 residual-correction 主线中最强的 Human 候选 |
| `qsh_rescorrnorm100_cap003_human_d128l3_e300_local` | 5 | `0.041517 ± 0.000202` | `0.114408 ± 0.000976` | cap=0.03 略差于上面版本 |
| `qsh_rescorrnorm100_gate_human_d128l3_e300_local` | 5 | `0.041532 ± 0.000183` | `0.114466 ± 0.001029` | confidence gate 未带来改善 |
| `qsh_rescorrnorm100_selfgate_human_d128l3_e300_local` | 5 | `0.041475 ± 0.000215` | `0.114349 ± 0.000849` | self gate 与当前最强版本基本持平 |

### 2.2 `USHCN`

| 版本 | 轮数 | MSE 均值 ± std | MAE 均值 ± std | 结论 |
|------|------|----------------|----------------|------|
| `qsh_rescorrnorm100_cap003_ushcn_itr5_local` | 5 | `0.179802 ± 0.036938` | `0.262676 ± 0.013521` | 当前 residual-correction 主线的直接起点；4 个好轮很强，但有 1 个明显坏轮 |
| `qsh_rescorrnorm100_cap003_epcap5_ushcn_itr5_local` | 5 | `0.192962 ± 0.029346` | - | `event_proj` norm cap=5 失败 |
| `qsh_rescorrnorm100_cap003_epcap6_ushcn_itr5_local` | 5 | `0.214128 ± 0.042746` | - | `event_proj` norm cap=6 明显失败 |
| `qsh_rescorrnorm100_cap003_selfgate_ushcn_itr5_local` | 5 | `0.190852 ± 0.037656` | `0.264263 ± 0.010909` | self-gated residual correction 失败 |
| `qsh_rescorrnorm100_cap003_confgate_ushcn_itr5_local` | 5 | `0.207944 ± 0.039097` | `0.272848 ± 0.010799` | confidence-gated residual correction 明显失败 |
| `qsh_cap003_budget075_ushcn_itr5_local` | 5 | `0.194180 ± 0.036793` | `0.269898 ± 0.014572` | 统一收紧共享创新预算失败 |

## 3. 最近一轮单因素试验链的结论

### 3.1 `event_proj` 有效范数裁剪

| 版本 | 结果 | 结论 |
|------|------|------|
| `epcap5` | `0.192962 ± 0.029346` | 伤到好轮，未解坏轮 |
| `epcap6` | `0.214128 ± 0.042746` | 明显更差 |

结论：

- `event_proj` 权重范数不是当前坏轮的根因级解释；
- 简单裁剪只会误伤有效 seed。

### 3.2 输出端 residual correction 门控

| 版本 | 数据集 | 结果 | 结论 |
|------|--------|------|------|
| `selfgate` | `HumanActivity` | `0.041475 ± 0.000215` | 仅微小变化 |
| `selfgate` | `USHCN` | `0.190852 ± 0.037656` | 失败 |
| `confgate` | `USHCN` | `0.207944 ± 0.039097` | 明显失败 |

结论：

- 问题不在输出 residual correction 没有被正确 gate；
- 输出端再叠 gate 不能解决 `USHCN` 坏轮。

### 3.3 统一收紧共享创新预算

| 版本 | 数据集 | 结果 | 结论 |
|------|--------|------|------|
| `budget075` | `USHCN` | `0.194180 ± 0.036793` | 失败 |

结论：

- 同时收紧 event 注入与 quaternion residual，会误伤好轮；
- 但照样挡不住坏轮。

### 3.4 `routebudget` 本地扫描补记

| 版本 | 数据集 | 结果 | 结论 |
|------|--------|------|------|
| `qsh_routebudget_floor025_human_local` | `HumanActivity` | `0.041582 ± 0.000190` | Human 侧稳定可用 |
| `qsh_routebudget_floor025_ushcn_local` | `USHCN` | `0.169866 ± 0.014330` | 当前这条链里最可接受的 USHCN 候选 |
| `qsh_routebudget_retainbudget_human_local` | `HumanActivity` | `0.041596 ± 0.000183` | 与 `floor025` 基本持平 |
| `qsh_routebudget_retainbudget_ushcn_local` | `USHCN` | `0.190289 ± 0.029010` | 明显劣于 `floor025` |
| `qsh_routebudget_floor0_human_d128l3_local` | `HumanActivity` | `0.041557 ± 0.000254` | Human-only 扫描中的最优邻近点 |

结论：

- `routebudget` 方向在 `HumanActivity` 上大多集中在 `0.04156 ~ 0.04160`，说明它更像稳态微调，而不是大幅改进源；
- `floor025` 比 `retainbudget` 明显更合理；
- 但这条链还没有完成 `P12 / MIMIC_III / MIMIC_IV` 的正式扩展验证。

### 3.5 `outputfusion / coredecoderinit` 链补记

| 版本 | 数据集 | 结果 | 结论 |
|------|--------|------|------|
| `qsh_outputfusion005_coredecoderinit_human_local` | `HumanActivity` | `0.041545 ± 0.000235` | Human 上可用 |
| `qsh_outputfusion005_coredecoderinit_ushcn_local` | `USHCN` | `0.223903 ± 0.049122` | USHCN 明显失败 |
| `qsh_coredecoder_rescorr_main_local` | `HumanActivity` | `0.041477 ± 0.000186` | 当前 Human 单点结果很好 |
| `qsh_coredecoder_rescorr_main_local` | `USHCN` | `0.253231 ± 0.023152` | 本地已证伪，不适合直接上服务器 |
| `qsh_besthuman_decoderinit_local` | `HumanActivity` | `0.041692 ± 0.000392` | 只在 Human 上有边际意义 |

结论：

- `HyperIMTS` 一致初始化与输出融合，单看 `HumanActivity` 可以做出不错数值；
- 但它们当前无法守住 `USHCN`；
- 因此这条链目前不能被当作下一条默认服务器主线。

### 3.6 `spikeselectprop_adapt` 链补记

| 版本 | 数据集 | 结果 | 结论 |
|------|--------|------|------|
| `spikeselectprop_adapt025_qbias4_local` | `HumanActivity` | `0.041735 ± 0.000144` | 不伤 Human |
| `spikeselectprop_adapt025_qbias4_local` | `USHCN` | `0.164609 ± 0.002637` | 当前已回收本地结果里最强的传播选择近邻之一 |
| `spikeselectprop_adapt025_itr10` | `HumanActivity` | `0.041747 ± 0.000165` | 稳定 |
| `spikeselectprop_adapt025_itr10` | `USHCN` | `0.165544 ± 0.005667` | 与上面版本同量级，方差可接受 |
| `spikeselectprop_quat015_local` | `USHCN` | `0.175164 ± 0.019627` | 继续压 quaternion 比例后反而变差 |

结论：

- 自适应传播选择比最近这轮 residual-correction 输出头近邻更接近有效方向；
- `adapt025_qbias4` 与 `adapt025_itr10` 都已经达到「主体 seed 明显优于 HyperIMTS 参考」的本地水位；
- 但由于当前工作区主线后来转向 residual-correction，这条链此前没有被完整写回文档，这里补记保留。

## 4. 当前最重要的结果解读

1. 最近这轮没有找到一个能替代 `qsh_rescorrnorm100_cap003_ushcn_itr5_local` 的更优单因素改动。
2. `cap003` 仍然表现为：
- 4 个非常强的好轮；
- 少数坏轮把均值与方差拉坏。
3. 最近连续失败的三个方向说明：
- 不是输出 residual correction 没有 gate 好；
- 不是统一预算没有收紧好；
- 更像是活跃主路径的内部比例失衡。
4. 回补后的本地结果还说明：
- `spikeselectprop_adapt025` 与 `routebudget_floor025` 曾经给出过更强的 `USHCN` 局部结果；
- 但 `outputfusion / coredecoderinit` 这条「保主干再叠输出头」路线在本地 `USHCN` 上已经明显失败。

## 5. 当前结果评估口径

截至 2026-04-21，结果解释标准已经调整为：

1. 非 `USHCN` 数据集是当前主优化目标；
2. `USHCN` 不再要求优先压低最坏 seed 上界；
3. 对 `USHCN`，更重要的是：
- `itr=5` 里大部分 seed 优于 `HyperIMTS` 论文/项目参考；
- 不要为了少量坏轮而把整体结构改得过度保守；
4. 因此，后续实验会优先关注：
- 非 `USHCN` 数据集是否获得稳定增益；
- `USHCN` 是否仍保持“主体 seed 优于 HyperIMTS”的可接受水平。

## 6. 当前最值得保留的工作判断

1. 后续不应再把“压 `USHCN` 坏轮上界”作为唯一主线。
2. 需要优先寻找能改善非 `USHCN` 数据集、同时不明显破坏 `USHCN` 主体 seed 的结构改动。
3. 后续如果继续做结构试验，应更靠近：
- route / propagation 选择机制；
- quaternion 有效残差占比；
- 活跃主路径之间的内部比例约束；
而不是继续在输出头上做文章。
