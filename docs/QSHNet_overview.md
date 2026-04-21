# QSH-Net 当前总览

> **最后更新：** 2026-04-21
> **用途：** 一页式入口文档，用于快速回答「当前模型是什么、当前保留哪版、当前效果怎样、下一步从哪里继续」。

## 1. 一句话结论

当前 QSH-Net 更准确的表述是：

- `HyperIMTS` 超图主干；
- `retain` 调制的连续脉冲路由；
- 节点级 quaternion 残差增强；
- 已被唤醒、但仍不稳定的 event 注入；

而不应再表述为：

- `Spike + Event + Quaternion` 已经稳定充分协同。

## 2. 当前工作区主线是什么

当前直接工作的主线，是 `qsh_rescorrnorm100_cap003` 这一类 residual-correction 变体。

它的含义是：

- 保留 HyperIMTS 主干；
- 保留已唤醒的 event 注入；
- 保留 quaternion residual；
- 在输出端增加带归一化和上界的 residual correction；
- 当前关注点不是继续造新大版本，而是为跨数据集平均表现筛选更合理的结构改动。

## 3. 当前效果怎样

### 3.1 `HumanActivity`

当前本地最强 residual-correction 近邻结果：

- `qsh_rescorrnorm100_human_d128l3_e300_local`
- MSE `0.041469 ± 0.000171`

与 HyperIMTS 论文/项目参考对比：

- HyperIMTS: `0.0421 ± 0.0021`

解读：

- 已经非常接近；
- 它更适合作为本地快速结构筛选代理集；
- 但单独继续压它本身的边际收益有限。

### 3.2 `USHCN`

当前直接起点：

- `qsh_rescorrnorm100_cap003_ushcn_itr5_local`
- MSE `0.179802 ± 0.036938`

细看单轮：

- `0.162456, 0.163454, 0.160893, 0.253606, 0.158601`

解读：

- 4 个 seed 很强；
- 1 个明显坏轮把均值与方差拉坏；
- 当前更合理的要求是：`itr=5` 中大部分 seed 优于 `HyperIMTS` 论文/项目参考；
- 不再为了极少数坏轮而默认做更强保守化。

## 4. 最近为什么没有继续在输出头上优化

因为最近刚完成的 3 条单因素已经说明，那不是根因级修复：

1. `self-gated residual correction`
- `USHCN = 0.190852 ± 0.037656`
- 失败

2. `confidence-gated residual correction`
- `USHCN = 0.207944 ± 0.039097`
- 更差

3. `innovation_budget_init = 0.75`
- `USHCN = 0.194180 ± 0.036793`
- 失败

这说明：

- 问题不在输出 residual correction 没有门控好；
- 也不在统一收紧 event / quaternion 共享预算；
- 更像是活跃主路径的内部比例失衡。

## 5. 当前最重要的判断

1. 当前主优化目标应转向非 `USHCN` 数据集的稳定收益。
2. `HumanActivity` 更适合作为本地快速筛选代理集，而不是唯一优化终点。
3. `USHCN` 现在是约束项：要求 `itr=5` 中大部分 seed 优于 `HyperIMTS` 论文/项目参考，而不是执着于最坏轮上界。
4. 输出端再打补丁没有解决问题。
5. 后续应更靠近 route / propagation / quaternion 有效残差占比这些根因级位置继续做结构试验。

## 6. 详细文档入口

- [QSHNet_architecture_status.md](/opt/Codes/PyOmniTS/docs/QSHNet_architecture_status.md)
- [QSHNet_results_summary.md](/opt/Codes/PyOmniTS/docs/QSHNet_results_summary.md)
- [hyperparameter_tuning_plan.md](/opt/Codes/PyOmniTS/docs/hyperparameter_tuning_plan.md)
- [QSHNet_evolution.md](/opt/Codes/PyOmniTS/docs/QSHNet_evolution.md)
- [QSHNet_server_validation.md](/opt/Codes/PyOmniTS/docs/QSHNet_server_validation.md)

## 7. 当前已做过但需要记住的补充实验

最近补回文档后，当前应额外记住 3 条已经实际跑过的分支：

1. `routebudget`
- `qsh_routebudget_floor025_ushcn_local = 0.169866 ± 0.014330`
- 说明 route budget 这条链曾给出过可接受的 `USHCN` 本地结果。

2. `spikeselectprop_adapt`
- `spikeselectprop_adapt025_qbias4_local = 0.164609 ± 0.002637`
- `spikeselectprop_adapt025_itr10 = 0.165544 ± 0.005667`
- 说明传播选择自适应链不是空想分支，而是已经有过较强本地结果。

3. `coredecoderinit + residual correction`
- `qsh_coredecoder_rescorr_main_local` 在 `HumanActivity` 上是 `0.041477 ± 0.000186`
- 但在 `USHCN` 上退化到 `0.253231 ± 0.023152`
- 因此这条链目前不能再写成“默认下一条主线”，只能记为已做过且本地 `USHCN` 未通过的候选。
