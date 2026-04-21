# QSH-Net 服务器验证说明

> **最后更新：** 2026-04-21
> **用途：** 说明当前推荐的服务器验证对象、运行顺序、脚本入口与结果回收方式。

## 1. 当前推荐验证对象

当前不推荐立刻把最近这轮 `residual-correction` 近邻版本直接放到服务器做全数据集验证。

原因：

- 最近连续 3 条单因素都没有在本地 `USHCN` 上通过筛选；
- 失败版本包括：
  - `qsh_rescorrnorm100_cap003_selfgate_ushcn_itr5_local`
  - `qsh_rescorrnorm100_cap003_confgate_ushcn_itr5_local`
  - `qsh_cap003_budget075_ushcn_itr5_local`
- 在这种情况下，把它们直接扩展到服务器全数据集验证没有价值。

## 2. 最近一轮本地筛选结论

### 2.1 `selfgate`

- `USHCN = 0.190852 ± 0.037656`
- 失败

### 2.2 `confgate`

- `USHCN = 0.207944 ± 0.039097`
- 明显失败

### 2.3 `budget075`

- `USHCN = 0.194180 ± 0.036793`
- 失败

当前判断：

- 这 3 个版本都不值得进入服务器全数据集验证名单；
- 当前服务器验证仍应优先保留已经完成正式回收的历史候选，如 `spikeselectprop_res005`，而不是把这批失败近邻再扩展出去。

## 3. 服务器验证的使用原则

当前阶段默认遵守：

1. 本地如果只是在 `USHCN` 上压坏轮、但其他数据集没有明确收益，不优先上服务器；
2. 本地如果只是 `HumanActivity` 有轻微变化，也不上服务器；
3. `USHCN` 的本地筛选标准改为：
- `itr=5` 中大部分 seed 优于 `HyperIMTS` 论文/项目参考；
- 而不是必须先把最坏轮压到很低；
4. 服务器优先用于：
- 已通过本地双数据集或至少 `USHCN` 本地筛选的候选；
- 正式论文候选版本；
- 多数据集平均表现验证。

## 4. 当前服务器正式参考结果

当前可继续引用的正式服务器结果，仍以历史候选为主：

### 4.1 `spikeselectprop_res005`

| 数据集 | 轮数 | QSHNet MSE | 结论 |
|------|------|------------|------|
| `USHCN` | 5 主表 / 10 压测 | `0.16750 ± 0.00357` | 当前历史论文候选中的最强结果之一 |
| `HumanActivity` | 5 | `0.04172 ± 0.00018` | 接近 HyperIMTS 论文/项目参考 |
| `P12` | 5 | `0.30087 ± 0.00084` | 小幅退化但稳定 |
| `MIMIC_III` | 5 | `0.39396 ± 0.00308` | 基本持平 |
| `MIMIC_IV` | 5 | `0.21549 ± 0.00188` | 暂按旧参考值略优 |

### 4.2 `eventdensvar_main`

| 数据集 | 轮数 | MSE 均值 ± std | 结论 |
|------|------|----------------|------|
| `HumanActivity` | 5 | `0.04174 ± 0.00019` | 改善可复现 |
| `USHCN` | 10 | `0.1886 ± 0.0324` | 坏轮明显，不能升级为主线 |
| `P12` | 5 | `0.30092 ± 0.00062` | 稳定 |
| `MIMIC_III` | 5 | `0.39791 ± 0.01530` | 均值可接受，但仍有单轮失稳 |

## 5. 当前推荐做法

如果后续要再上服务器，建议先满足下面至少一条：

1. 非 `USHCN` 数据集上已有明确收益，并且 `USHCN itr=5` 主体 seed 仍优于 `HyperIMTS` 论文/项目参考；
2. 本地至少证明该版本不是单纯“为救 USHCN 坏轮而保守化”的近邻；
3. 本地 `HumanActivity` 没有出现明显退化；
4. 结构上属于新的论文候选主线，而不是仅仅一个失败近邻。

在达到这些条件之前，服务器资源优先留给：

- 正式主线的复验；
- 历史结果补表；
- 其他数据集的完整对照。

## 6. 补记：已准备但当前不推荐上服务器的候选

### 6.1 `qsh_coredecoder_rescorr_main`

这条链已经做过本地筛查：

- `HumanActivity = 0.041477 ± 0.000186`
- `USHCN = 0.253231 ± 0.023152`

结论：

- 它在 `HumanActivity` 上很好；
- 但在 `USHCN` 上本地已经明显失败；
- 因此当前不应再把它写成“默认下一条服务器主线”。

对应脚本仍可作为历史草稿保留：

- [server_validate_coredecoder_rescorr_all.sh](/opt/Codes/PyOmniTS/scripts/QSHNet/server_validate_coredecoder_rescorr_all.sh)
- [server_validate_coredecoder_rescorr_mimic_iv.sh](/opt/Codes/PyOmniTS/scripts/QSHNet/server_validate_coredecoder_rescorr_mimic_iv.sh)

### 6.2 如果后续还要新增服务器候选，优先看什么

从当前已经补回的本地结果看，更接近服务器候选门槛的是：

1. `spikeselectprop_adapt025_qbias4_local`
- `HumanActivity = 0.041735 ± 0.000144`
- `USHCN = 0.164609 ± 0.002637`

2. `qsh_routebudget_floor025_ushcn_local`
- `USHCN = 0.169866 ± 0.014330`

但它们仍缺：

- `P12 / MIMIC_III / MIMIC_IV` 的本地或服务器多数据集回收；
- 更完整的正式候选叙事整理。
