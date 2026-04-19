# QSH-Net 超参数与诊断计划

> **最后更新：** 2026-04-19

## 当前状态

当前工作重点已经从「常规超参数扫描」切换为「结构诊断优先」，并且已经完成一轮围绕 `event` / `retain` / `quaternion` 的结构性试验链。

当前最重要的现实结论不是某个学习率或 batch size 没调好，而是：

1. 旧 M1 的真实训练动力学并不是「Spike + Event + Quaternion 协同」。
2. 旧 M1 更接近 `HyperIMTS + retain 调制 + quaternion 残差`。
3. 如果希望保住论文里的「三元素统一框架」，必须先让 `event` 分支变成真实可训练、真实参与训练的分支，而不是继续把它当成挂名模块。

同时，`eventdensvar_main` 的服务器四数据集正式验证已经完成，当前需要以该结果为准，而不能再只依据本地 `USHCN itr=5` 的筛选结果做判断。

2026-04-19 之后又补做了三类单因素稳定化试验：

- `routebound075`: 单纯压 `route_logit`，失败；
- `eventprojgrad2`: 单纯加速 `event_proj` 学习，失败；
- `routeconfvar`: route dispersion 感知的 variable event 衰减，失败。
- `memgradclip012`: 单纯裁剪 `membrane_proj` 梯度，失败。
- `routecenter`: route logit 样本内均值中心化，失败。
- `eventgateconst_quat020`: 单纯压 quaternion residual cap，不是根因级修复，失败后回退。
- `eventgateconst_meminit001`: 主动提高 `membrane` 初始 route dispersion，明显失败后回退。
- `eventgateconst_routedetach`: 切断 route density 对稳定化支路的反向梯度，失败后回退。

因此后续不应继续把问题简化为“某个 gate 太大/太小”“event projection 学得不够快”“membrane 梯度尖峰需要直接裁剪”或“route mean 漂移需要中心化”。
同样也不应再默认尝试：

- 修改 `membrane` 初始化来人为制造早期 route dispersion
- 把 route density 从稳定化训练反馈里硬切出去
- 继续压单纯 quaternion residual cap

## 当前主线候选结果（更新到 2026-04-18）

| 版本 | 数据集 | 轮数 | MSE 均值 ± std | 结论 |
|------|--------|------|----------------|------|
| `retaincap_main` | USHCN | 5 | **0.1673 ± 0.0033** | 方向 B 的最稳基线，说明 `retain` 约束有效 |
| `retaincap_quatbound` | USHCN | 5 | **0.1674 ± 0.0093** | quaternion 残差约束没有带来额外收益 |
| `eventfusion_sigmoid` | HumanActivity | 3 | **0.0426 ± 0.0009** | `event` 活化后在简单数据上稳定 |
| `eventfusion_sigmoid` | USHCN | 5 | **0.1803 ± 0.0386** | `event` 活化成功，但高方差尾部明显放大 |
| `eventnorm_main` | HumanActivity | 3 | **0.0430 ± 0.0013** | 加入 `event` 独立归一化后仍然稳定 |
| `eventnorm_main` | USHCN | 5 | **0.1653 ± 0.0011** | 当前最有价值的三元素框架候选，短重复下兼顾激活与稳定 |
| `eventgain_main` | HumanActivity | 3 | **0.0424 ± 0.0006** | 主干条件化 gain 在简单数据上不坏 |
| `eventgain_main` | USHCN | 5 | **0.1974 ± 0.0633** | 条件化 gain 明显放大坏轮，方向失败 |
| `event_temporal_only` | HumanActivity | 3 | **0.0422 ± 0.0005** | temporal-only 不影响简单数据稳定性 |
| `event_temporal_only` | USHCN | 5 | **0.1779 ± 0.0365** | 只保留 temporal 注入不如双支路 `eventnorm_main` |
| `eventnorm_clip` | HumanActivity | 3 | **0.0422 ± 0.0005** | `tanh` 裁剪不影响简单数据 |
| `eventnorm_clip` | USHCN | 5 | **0.1898 ± 0.0381** | `tanh` 裁剪把结果拉成两极分化，方向失败 |
| `eventscalecap_main` | HumanActivity | 3 | **0.0430 ± 0.0013** | 加入 `event_scale` 温和上界后总体保持稳定 |
| `eventscalecap_main` | USHCN | 5 | **0.1663 ± 0.0027** | 与 `eventnorm_main` 接近，说明温和上界是安全保险丝 |
| `eventnorm_itr10` | USHCN | 10 | **0.1829 ± 0.0279** | 长重复下仍有尾部风险，不能直接视为最终定版 |
| `eventscalecap_itr10` | USHCN | 10 | **0.1728 ± 0.0222** | 明显优于 `eventnorm_itr10`，但仍保留尾部坏轮 |
| `eventrescap_main` | HumanActivity | 3 | **0.04185 ± 0.00009** | HumanActivity 改善，但不足以证明方向成立 |
| `eventrescap_main` | USHCN | 5 | **0.1834 ± 0.0273** | 相对主线明显退化，方向否决 |
| `eventdenscap_main` | HumanActivity | 3 | **0.04181 ± 0.00011** | HumanActivity 改善保持 |
| `eventdenscap_main` | USHCN | 5 | **0.1891 ± 0.0347** | 全路径 density 抑制明显更差，方向否决 |
| `eventdensvar_main` | HumanActivity | 3 | **0.04181 ± 0.00011** | 保持 HumanActivity 改善 |
| `eventdensvar_main` | USHCN | 5 | **0.1703 ± 0.0058** | 略弱于主线，但方差收敛明显，当前可接受保留候选 |
| `eventdensvar_main` | HumanActivity | 5 | **0.04174 ± 0.00019** | 服务器验证确认改善稳定可复现 |
| `eventdensvar_main` | USHCN | 10 | **0.1886 ± 0.0324** | 服务器验证显示坏轮仍明显，不能升级为统一主线 |
| `eventdensvar_main` | P12 | 5 | **0.30092 ± 0.00062** | 服务器验证稳定，通过跨数据集可用性检查 |
| `eventdensvar_main` | MIMIC_III | 5 | **0.39791 ± 0.01530** | 均值可接受，但仍有单轮失稳 |
| `variable residual only + adaptive fused-cap` | USHCN | 5 | **0.16785 ± 0.00772** | 当前代码主线，用户已接受 |
| `variable residual only + adaptive fused-cap` | USHCN | 10 | **0.19035 ± 0.03078** | 诊断运行显示后段坏轮仍存在 |
| `routebound075` | USHCN | 10 | **0.18132 ± 0.02602** | 单纯压 route logit 失败 |
| `eventprojgrad2` | USHCN | 10 | **0.18962 ± 0.03279** | 加速 event projection 学习失败 |
| `routeconfvar` | USHCN | 10 | **0.19155 ± 0.03720** | route dispersion 感知 event 衰减失败 |
| `memgradclip012` | USHCN | 10 | **0.17775 ± 0.02485** | 均值尚可，但坏轮上界仍高，已撤回 |
| `routecenter` | USHCN | 10 | **0.18433 ± 0.03669** | route 均值中心化失败，已撤回 |
| `eventgateconst` | USHCN | 10 | **0.17587 ± 0.01571** | 当前唯一值得保留的尾部压制候选 |
| `eventgateconst_quat020` | USHCN | 10 | **0.17550 ± 0.01473** | 数值略优，但 `quat_clip=0.0`，不是根因级改动，已撤回 |
| `eventgateconst_meminit001` | USHCN | 10 | **0.21388 ± 0.08635** | 主动提高 route 初始分散度明显失败 |
| `eventgateconst_routedetach` | USHCN | 10 | **0.18122 ± 0.02408** | 切断 route density 反向梯度失败 |

## 当前正式结论

在服务器四数据集验证之后，当前最准确的结论应当是：

1. `eventscalecap_main / eventscalecap_itr10` 仍然是统一主线母体。
2. `eventdensvar_main` 仍值得保留，但只能作为 `event density` 方向的工作区候选，不能升级为新的统一主线。
3. `HumanActivity` 与 `P12` 已经通过，`MIMIC_III` 基本可接受但仍有坏轮。
4. 当前真正没有过关的数据集是 `USHCN`，因此后续优化目标必须聚焦到坏轮压制，而不是继续泛泛地“调参”。

## 当前训练配置

| 参数 | P12 | HumanActivity | USHCN |
|------|-----|---------------|-------|
| d_model | 256 | 128 | 256 |
| n_layers | 2 | 3 | 1 |
| n_heads | 8 | 1 | 1 |
| batch_size | 32 | 32 | 16 |
| learning_rate | 1e-3 | 1e-3 | 1e-3 |
| lr_scheduler | DelayedStepDecayLR | DelayedStepDecayLR | DelayedStepDecayLR |
| train_epochs | 300 | 300 | 300 |
| patience | 10 | 10 | 10 |
| optimizer | AdamW | AdamW | AdamW |
| weight_decay | 1e-4 | 1e-4 | 1e-4 |

## 已确认有效的历史改动

### 1. 恒等/安全初始化

| 组件 | 初始化策略 | 作用 |
|------|------------|------|
| `QuaternionLinear` | `r=I, i=j=k=0` | 避免随机噪声破坏主干 |
| `SpikeSelection/SpikeRouter.membrane_proj` | `weight=0, bias=0` | 保证初始接近恒等 |

### 2. AdamW + Weight Decay

| 配置 | 效果 |
|------|------|
| AdamW, `weight_decay=1e-4` | 明显优于旧的 Adam 无衰减配置，尤其降低了 USHCN 的方差 |

## 已完成但已否决的试验

### bounded-gate 保守稳定化（已回退）

| 数据集 | 轮数 | MSE 均值 ± std | 结论 |
|--------|------|----------------|------|
| USHCN | 10 | **0.2068 ± 0.0375** | 明显退化，不能保留 |
| P12 | 5 | **0.3012 ± 0.0012** | 基本不变 |
| MIMIC_III | 5 | **0.3983 ± 0.0066** | 有小幅改善，但不具备统一推广性 |

结论：该方案具有明显数据集依赖性，已整套回退，不再作为默认方向。

## 本地参数诊断结论

### 诊断方法

- 在 `exp/exp_main.py` 中新增 epoch 级 `QSHDiag` 日志。
- 对 `USHCN` 与 `HumanActivity` 做短程本地训练，并读取最终 checkpoint 参数。

### 诊断结论

1. **旧 M1 中的 event residual 分支最初是死分支。**
   - `event_proj.weight norm = 0.0`
   - `event_residual_scale = 0.0`
   - 以上现象在 USHCN 与 HumanActivity 上都一致出现。

2. **真正持续学习的是 retain 与 quaternion。**
   - `retain_log_scale` 会增长。
   - `membrane_proj.weight` 会偏离初始零值。
   - `quat_gate.weight` 与 `quat_h2n` 会持续增长。

3. **USHCN 的不稳定更像是活跃分支被高方差数据放大。**
   - USHCN 上 `retain` / `membrane` / `quaternion` 的增长明显强于 HumanActivity。
   - 但两者都没有真正激活 event residual 分支。

## 结构试验链结论（2026-04-16）

### 方向 B：先承认真实活跃结构是 `retain + quaternion`

1. **`retaincap_main` 证明 `retain` 约束是有效的。**
   - `USHCN itr=5` 达到 `0.1673 ± 0.0033`。
   - 说明旧 M1 的坏轮确实与活跃分支放大有关。

2. **`retaincap_quatbound` 证明简单限制 quaternion residual 不是主要增益点。**
   - `USHCN itr=5` 为 `0.1674 ± 0.0093`。
   - 均值几乎不动，方差反而略有放大。

### 方向 A：围绕论文叙事激活 `event` 分支

1. **`eventfusion_sigmoid` 首次让 `event` 分支真正参与训练。**
   - `event_residual_scale` 不再是近零残差语义，而是稳定停在 `sigmoid ≈ 0.1`。
   - `event_proj.weight norm` 在 HumanActivity 与 USHCN 上都持续偏离 0。
   - 但 `USHCN itr=5` 退化到 `0.1803 ± 0.0386`，说明单纯增强融合强度会重新放大尾部风险。

2. **`eventnorm_main` 证明“活着的 event 分支 + 独立归一化”是更合理的统一框架版本。**
   - 在保持 `event` 持续参与训练的同时，`USHCN itr=5` 回到 `0.1653 ± 0.0011`。
   - 这说明 `event` 支路的问题不只是“强度太弱”，还包括“缺少独立控制”。

3. **`eventgain_main` 否定了“主干条件化 gain”这一方向。**
   - `HumanActivity itr=3` 为 `0.0424 ± 0.0006`。
   - 但 `USHCN itr=5` 退化到 `0.1974 ± 0.0633`，并重新出现重度坏轮。
   - 这说明 `event` 支路的问题不是“还缺一层主干条件化控制器”。

4. **`event_temporal_only` 否定了“只保留 temporal 注入就会更稳”的猜想。**
   - `HumanActivity itr=3` 为 `0.0422 ± 0.0005`。
   - 但 `USHCN itr=5` 仍为 `0.1779 ± 0.0365`，明显不如双支路 `eventnorm_main`。
   - 这说明双支路注入本身不是主要问题，关键仍是简单而独立的控制。

5. **`eventnorm_clip` 否定了“直接裁剪归一化后的 event_delta 幅度”这一方向。**
   - `HumanActivity itr=3` 为 `0.0422 ± 0.0005`。
   - 但 `USHCN itr=5` 退化到 `0.1898 ± 0.0381`，出现明显双峰坏轮。
   - 这说明问题不在于归一化后的 `event_delta` 还存在少量大幅值，而在于如何更平滑地控制总注入量。

6. **`eventscalecap_main` 证明“只限制 event 总注入量”是安全的，但短重复增益有限。**
   - `event_scale` 从 `sigmoid(event_residual_scale)` 改为 `clamp(sigmoid(...), max=0.12)`。
   - `USHCN itr=5` 为 `0.1663 ± 0.0027`，和 `eventnorm_main` 非常接近。
   - 说明温和上界不会破坏当前三元素框架，但单看 `itr=5` 还称不上显著突破。

7. **`eventscalecap_itr10` 证明温和上界能改善长重复稳定性，但还不能宣布问题解决。**
   - `USHCN itr=10` 为 `0.1728 ± 0.0222`。
   - 相比 `eventnorm_itr10` 的 `0.1829 ± 0.0279`，均值和方差都更好。
   - 但仍然出现 `iter8 = 0.2355` 级别的坏轮，因此只能视为“有效改善”，不能视为最终工程定版。

8. **`eventnorm_itr10` 否定了“已经彻底稳定”的乐观判断。**
   - `USHCN itr=10` 为 `0.1829 ± 0.0279`。
   - 虽然明显优于 `eventfusion_sigmoid` 的长尾不稳定，但仍然出现 `iter5/7/8` 级别的坏轮。
   - 当前更准确的结论是：`eventnorm` 足以支撑三元素统一框架叙事，但还不能被视为最终工程定版。

9. **`eventrescap_main` 否定了“直接给 event residual 加相对主干范数上界”这一方向。**
   - `HumanActivity itr=3` 改善到 `0.04185 ± 0.00009`。
   - 但 `USHCN itr=5` 退化到 `0.1834 ± 0.0273`。
   - 这说明全局比例硬约束会把高方差数据上的有效 `event` 一起压掉。

10. **`eventdenscap_main` 否定了“对 temporal + variable 全路径同时做 density-aware 衰减”这一方向。**
   - `HumanActivity itr=3` 为 `0.04181 ± 0.00011`。
   - 但 `USHCN itr=5` 进一步退化到 `0.1891 ± 0.0347`。
   - 这说明 temporal 路径上的 density-aware 抑制不是安全保险丝，反而更像误伤有效注入。

11. **`eventdensvar_main` 证明如果要引入 density-aware 控制，只能收窄到 variable 路径。**
   - `HumanActivity itr=3` 维持在 `0.04181 ± 0.00011`。
   - `USHCN itr=5` 为 `0.1703 ± 0.0058`。
   - 相比 `eventscalecap_main` 的 `0.1663 ± 0.0027`，均值略差；
   - 但相比 `eventrescap_main` / `eventdenscap_main`，它已经显著收敛，且没有再出现严重长尾。
   - 当前更准确的定位是：它不是新的统一主线，但已经是可接受保留候选。

12. **服务器四数据集验证进一步收紧了 `eventdensvar_main` 的定位。**
   - `HumanActivity itr=5` 为 `0.04174 ± 0.00019`，改善稳定复现。
   - `P12 itr=5` 为 `0.30092 ± 0.00062`，结果稳定。
   - `MIMIC_III itr=5` 为 `0.39791 ± 0.01530`，均值可接受但仍有单轮失稳。
   - `USHCN itr=10` 退化到 `0.1886 ± 0.0324`，并重新出现明显坏轮。
   - 因此它只能保留为工作区候选，不能替代 `eventscalecap_main / eventscalecap_itr10`。

## 当前工作判断

### 现在不应该优先做的事

- 继续做常规学习率、batch size、层数扫描
- 把问题简单归结为 `event_gate` 没调好
- 在 USHCN 上只看 3~5 轮结果就下结论
- 继续无差别地加大 `event` 融合强度
- 把 `eventrescap_main` / `eventdenscap_main` 这种全局收缩方向继续细调

### 当前最重要的判断

1. **如果目标是追求最稳的短期结果，`retaincap_main` 仍然是最保守的稳定化版本。**
2. **如果目标是保住论文里的三元素统一框架，当前更合适的主线候选已经升级为 `eventnorm + mild event_scale cap`。**
3. **`eventdensvar_main` 可以作为当前可接受保留候选，但还不应替代统一母体。**
4. **`eventscalecap_itr10` 仍然保留尾部坏轮，当前更适合写成“可训练、可控、已明显改善但尚未彻底稳定”的阶段性结论。**
5. **服务器正式验证之后，下一步不该再追求继续改善 `HumanActivity`，而应只盯 `USHCN` 坏轮压制，同时守住 `P12 / MIMIC_III`。**
6. **最新失败试验说明，不应继续做单点 gate bound、单点梯度倍率、单点梯度裁剪、route 均值中心化或 route dispersion 触发的局部 event 衰减。**
7. **当前唯一相对收敛的新判断是：`event_gate` 不应继续直接依赖 `route_logit`。**
8. **但 route 本身也不能被粗暴改初始化或切断反馈；这两条已被新增试验证伪。**

## 2026-04-19 之后的最新收敛判断

### 已知最有价值的新增结论

1. `eventgateconst` 有效
- 它把坏轮尾部从此前一批失败近邻实验的 `0.23~0.27` 区间，压回到 `0.21558`
- 但仍没有达到 `<=0.18` 的目标

2. `quat020` 不是根因修复
- 虽然数值略有改善
- 但 `quat_clip=0.0`
- 说明“压输出 residual 比例”没有真正进入主矛盾

3. `meminit001` 明确证伪
- 提前制造 route dispersion 会把系统直接推向更差的高方差状态

4. `routedetach` 明确证伪
- route density 虽然与坏轮相关，但其训练反馈不能被简单切断

### 因此，下一步不应再做什么

- 不继续试 `membrane_proj` 初始化技巧
- 不继续试 route density detach
- 不继续试简单 quaternion residual cap
- 不回到常规大规模学习率 / batch size 扫描

### 下一步真正值得保留的问题

在当前证据下，后续结构试验应围绕：

- `eventgateconst` 这个候选框架继续做
- 重点看 quaternion 参数演化方式本身
- 而不是继续碰 route 前端或全局收缩开关

### 下一步建议

1. 继续以 `eventscalecap_main` / `eventscalecap_itr10` 作为当前统一框架候选版本。
2. 把 `variable residual only + adaptive fused-cap` 记为当前代码主线，用于后续 USHCN 坏轮压制的直接起点。
3. 把 `eventdensvar_main` 记为历史可接受保留候选，而不是继续默认升级它。
3. 暂不继续做大规模超参数扫描。
4. 后续结构试验若继续推进，只围绕 `USHCN` 坏轮压制做单因素消融，而不是回到全路径收缩。
5. 新试验必须同时检查：
   - `USHCN itr>=5`，更稳妥是 `itr=10`
   - `HumanActivity itr=3`
   - 通过后再看 `P12 / MIMIC_III`
6. 当前不再把学习率、batch size、层数扫描视为主优先级；除非出现新的明确证据，否则默认问题首先是结构问题，不是常规超参数问题。

## 运行与诊断方式

```bash
# USHCN 标准运行
python main.py --is_training 1 --collate_fn collate_fn --loss MSE \
    --d_model 256 --n_layers 1 --n_heads 1 \
    --dataset_root_path storage/datasets/USHCN \
    --model_id QSHNet --model_name QSHNet \
    --dataset_name USHCN --dataset_id USHCN \
    --features M --seq_len 150 --pred_len 3 \
    --enc_in 5 --dec_in 5 --c_out 5 \
    --train_epochs 300 --patience 10 --val_interval 1 \
    --itr 5 --batch_size 16 --learning_rate 1e-3

# HumanActivity 标准运行
python main.py --is_training 1 --collate_fn collate_fn --loss MSE \
    --d_model 128 --n_layers 3 --n_heads 1 \
    --dataset_root_path storage/datasets/HumanActivity \
    --model_id QSHNet --model_name QSHNet \
    --dataset_name HumanActivity --dataset_id HumanActivity \
    --features M --seq_len 3000 --pred_len 300 \
    --enc_in 12 --dec_in 12 --c_out 12 \
    --train_epochs 300 --patience 10 --val_interval 1 \
    --itr 3 --batch_size 32 --learning_rate 1e-3

# 本地短程参数诊断（示例）
python main.py --is_training 1 --collate_fn collate_fn --loss MSE \
    --d_model 256 --n_layers 1 --n_heads 1 \
    --dataset_root_path storage/datasets/USHCN \
    --model_id QSHNet_diag_trace --model_name QSHNet \
    --dataset_name USHCN --dataset_id USHCN \
    --features M --seq_len 150 --pred_len 3 \
    --enc_in 5 --dec_in 5 --c_out 5 \
    --train_epochs 3 --patience 3 --val_interval 1 \
    --itr 1 --batch_size 16 --learning_rate 1e-3 --num_workers 0
```

## 备注

- AdamW 仅对 `model_name` 包含 `QSH` 的模型启用
- 每轮实验只改一个点，避免把结构问题和超参数问题混在一起
- USHCN 需要 `itr >= 5`，更稳妥是 `itr = 10`
- HumanActivity 用 `itr = 3` 即可做快速对照
- 在已经确认 `event` 可以被唤醒之后，重点应转向“如何让它长期稳定”，而不是重新回到常规调参
- `event density` 方向已经形成新约束：若继续推进，只看 variable 路径，不再同时动 temporal 路径
- 服务器正式验证之后，`eventdensvar_main` 的身份应固定为“保留候选”，而不是“待升级新主线”

## EQHO 开发阶段补充（2026-04-17）

### 已完成的本地筛查

在 `eventscalecap` 母体上完成了 `EQHO A3` 的本地筛查：

| 配置 | 数据集 | 轮数 | MSE 均值 ± std | 结论 |
|------|--------|------|----------------|------|
| `QSHNet_EQHO_A3` | HumanActivity | 3 | `0.0418 ± 0.0002` | 稳定，可训练 |
| `QSHNet_EQHO_A3` | USHCN | 5 | `0.1979 ± 0.0310` | 明显退化，方向失败 |

### 关键诊断结论

1. `EQHO` 不是死结构。
   - `HumanActivity` 与 `USHCN` 上都能看到 `mix_coef / mix_gain / residual_norm_mean` 的真实变化。

2. `A3` 的问题不是“没有学起来”，而是“学得过强”。
   - `HumanActivity` 上仍然保持温和的 real-dominant refinement；
   - `USHCN` 上完整动态 `mix_coef` 会快速偏离 real-dominant 区间，并伴随 `gain_mean`、`summary_mean`、`residual_norm_mean` 明显抬升。

3. 因此 `EQHO A3` 的停机结论已经成立。
   - 不再把 `A3` 视作当前主线候选；
   - 也不再继续在它上面做常规超参数修补。

### 后续约束更新

- 当前主线仍是 `eventscalecap_main / eventscalecap_itr10`
- 若继续推进 `EQHO`，下一步应优先做受限动态化版本（可记为 `A2.5`）
- `A2.5` 的核心思想应是：
  - 保留 real-dominant 模板；
  - 仅允许 `mix_coef` 在安全边界内做小幅偏移；
  - 不再直接放开完整 `r / i / j / k` mixing

### `A2.5` 本地筛查结果（已完成）

在 `A3` 之后，已完成 `A2.5` 的本地对照：

| 配置 | 数据集 | 轮数 | MSE 均值 ± std | 结论 |
|------|--------|------|----------------|------|
| `QSHNet_EQHO_A25` | HumanActivity | 3 | `0.0418 ± 0.0002` | 安全，可训练，不伤简单数据 |
| `QSHNet_EQHO_A25` | USHCN | 5 | `0.2062 ± 0.0264` | 仍明显退化，不能入主线 |

### `A2.5` 带来的新结论

1. `A2.5` 成功验证了 `A3` 的一部分归因。
   - `mix_coef` 被限制在 real-dominant 安全区之后，`A3` 中那种完全偏离实部主导的失稳不再出现。

2. 但 `A2.5` 没有把 `USHCN` 拉回主线水位。
   - `USHCN itr=5` 仍为 `0.2062 ± 0.0264`；
   - 甚至差于 `A3` 的 `0.1979 ± 0.0310`。

3. 因此 `EQHO` 当前的主要风险判断需要更新为：
   - 风险不再主要是 `mix_coef`；
   - 更可能是 temporal 分支的 `mix_gain` 在高方差数据上持续接近饱和，导致 hyperedge quaternion refinement 仍被过强放大。

### `EQHO` 后续唯一优先方向

如果后续继续推进 `EQHO`，不要再重复：

- 继续放宽 `mix_coef`
- 或继续只围绕 `mix_coef` 做边界修补

下一轮唯一最值得验证的核心假设应是：

- 在保持 `A2.5` 的 real-dominant 模板与 safety floor 完全不变的前提下，
- 只修改 `mix_gain` 的参数化或上界，
- 验证 `USHCN` 坏轮是否能被进一步压回。

### `A2.6` 本地筛查结果（已完成）

按上面的方向，已完成 `mix_gain hard cap` 对照：

| 配置 | 数据集 | 轮数 | MSE 均值 ± std | 结论 |
|------|--------|------|----------------|------|
| `QSHNet_EQHO_A26` | HumanActivity | 3 | `0.0418 ± 0.0003` | 与 `A2.5` 持平，说明 hard cap 不伤简单数据 |
| `QSHNet_EQHO_A26` | USHCN | 5 | `0.1846 ± 0.0297` | 明显优于 `A25/A3`，但仍弱于主线 |

### `A2.6` 带来的新结论

1. `mix_gain hard cap` 确实有效。
   - `USHCN` 上从 `A2.5` 的 `0.2062 ± 0.0264` 改善到 `0.1846 ± 0.0297`；
   - 也优于 `A3` 的 `0.1979 ± 0.0310`。

2. 但它仍然不是最终解。
   - 仍明显差于 `eventscalecap_main` 的 `0.1663 ± 0.0027`；
   - 且方差仍不够理想。

3. 当前 `EQHO` 的问题层级应进一步上移：
   - 不是只看 `mix_coef`
   - 也不是只看 `mix_gain`
   - 更可能是 `event_summary` 的幅值和统计形态本身，在高方差数据上仍会把后续分支推到高能区

### `EQHO` 下一轮若继续的唯一合理方向

如果后续还要继续推进 `EQHO`，不再优先做：

- 继续压 `mix_coef` 边界
- 继续仅调 `mix_gain_max`

下一轮更值得验证的是：

- `event_summary` 的归一化、幅值控制或统计重参数化

而不是再单纯修补它后面的 `coef` / `gain` 投影头。

### `S1` 本地筛查结果（已完成）

按上面的方向，已完成 `event_summary` 输出端归一化对照：

| 配置 | 数据集 | 轮数 | MSE 均值 ± std | 结论 |
|------|--------|------|----------------|------|
| `QSHNet_EQHO_S1` | HumanActivity | 3 | `0.0418 ± 0.0003` | 与 `A2.6` 基本持平，不伤简单数据 |
| `QSHNet_EQHO_S1` | USHCN | 5 | `0.2072 ± 0.0461` | 虽压低 `summary` 统计量，但结果明显退化 |

### `S1` 带来的新结论

1. `event_summary` 输出端的 `LayerNorm` 能明显压低诊断统计量。
   - `USHCN` 首轮 `QSHDiag` 中，temporal / variable `summary_mean` 大约只有 `0.20 / 0.22`；
   - 与 `A2.6` 在 `USHCN` 上 `5 ~ 11` 的量级相比，输出尺度被大幅压平。

2. 但这并没有带来更好的 `USHCN` 最终结果。
   - `USHCN itr=5` 退化到 `0.2072 ± 0.0461`；
   - 比 `A2.6` 的 `0.1846 ± 0.0297` 更差；
   - 也差于 `A3` 的 `0.1979 ± 0.0310`。

3. 因此当前可以排除一个简单假设：
   - 问题不是“只要把 `event_summary` 输出幅值压下来就能修复 `USHCN`”
   - 更深层的问题更可能在 `event_summary` 的信息组织方式，而不是单纯尺度大小

### `S1` 后的约束更新

如果后续还要继续推进 `EQHO`，不再优先做：

- 继续仅调 `mix_gain_max`
- 继续仅做 summarizer 输出端归一化

更值得看的将是：

- `event_summary` 的内部构造方式
- `main/event/gate` 三路信息在 summarizer 内部的融合形式

### `S2` 本地筛查结果（已完成）

按上面的方向，已完成 summarizer 分路静态聚合对照：

| 配置 | 数据集 | 轮数 | MSE 均值 ± std | 结论 |
|------|--------|------|----------------|------|
| `QSHNet_EQHO_S2` | HumanActivity | 3 | `0.0422 ± 0.0002` | 已伤简单数据，不如 `A2.6/S1` |
| `QSHNet_EQHO_S2` | USHCN | 5 | `0.2218 ± 0.0336` | 明显更差，方向失败 |

### `S2` 带来的新结论

1. 把 summarizer 改成“分路投影 + 静态加权聚合”并不能修复 `EQHO`。
2. `USHCN` 首轮诊断里，variable 支路 `coef_r` 已掉到 `0.56 ~ 0.61` 附近，说明这种改法会破坏 real-dominant 安全结构。
3. 因此不能把 summarizer 的局部拓扑重写视为有效出路。

### `S3` 本地筛查结果（已完成）

随后又完成了更保守的 residual-style summarizer 对照：

| 配置 | 数据集 | 轮数 | MSE 均值 ± std | 结论 |
|------|--------|------|----------------|------|
| `QSHNet_EQHO_S3` | HumanActivity | 3 | `0.0418 ± 0.0004` | 回到安全线附近，不伤简单数据 |
| `QSHNet_EQHO_S3` | USHCN | 5 | `0.2173 ± 0.0340` | 仍明显退化，方向失败 |

### `S3` 带来的新结论

1. residual-style summarizer 比 `S2` 稳定，但仍无法修复 `USHCN`。
2. 它说明“把 summarizer 改成主干主导的小残差结构”仍不足以解决 `EQHO` 的根本问题。
3. 因此当前 `EQHO` 最优探索版本仍然是 `A2.6`，而不是任何 `S*` 版本。

### `S2/S3` 后的约束更新

如果后续还要继续推进 `EQHO`，不再优先做：

- summarizer 输出端归一化
- summarizer 分路静态聚合
- summarizer residual-style 小修

当前更合理的策略是：

- 暂时冻结 `EQHO` 在 `A2.6`
- 若要继续改，应重新提出更上层的结构假设，而不是继续在 summarizer 层修补

## 2026-04-19：传播选择版 spike 的最新约束

为兼顾论文叙事，曾尝试把 spike 从「event 幅值控制器」改成「传播选择器」：

- 新增 `selection_weight = sigmoid(route_logit)`；
- 将 `obs_base` 改为 `obs_selected = obs_base * selection_weight`；
- 用 `obs_selected` 进入 temporal / variable node-to-hyperedge 传播；
- 其余 `eventgateconst`、event residual、fused-cap、quaternion 均保持不变。

本地 `USHCN itr=10` 结果：

| 配置 | 数据集 | 轮数 | MSE 均值 ± std | max | 结论 |
|------|--------|------|----------------|-----|------|
| `spikeselectprop_a1` | USHCN | 10 | `0.19942 ± 0.02713` | `0.26434` | 明显退化，失败 |

由此更新后续约束：

1. 不再把「传播选择」实现为直接缩放 observation message。
2. 不把 `selection_weight` 版 n2h 输入缩放作为论文主线。
3. 当前代码中的 `selection_mean / selection_std` 可保留为诊断字段，但不能据此认定传播选择结构有效。
4. 如果后续继续做 spike 叙事友好的结构，应优先考虑：
   - attention-level bias；
   - residual-style propagation selection；
   - 或只作为解释性 route diagnostic，而不是直接改变主传播输入。
5. 工程稳定主线仍应回到 `eventgateconst`，不要继续沿 `spikeselectprop_a1` 做参数微调。

### 后续修正：identity-preserving residual selection

`spikeselectprop_a1` 失败后，进一步确认其核心问题是破坏 identity initialization：

- 初始 `route_logit = 0`
- `selection_weight = sigmoid(0) = 0.5`
- 直接乘到 n2h observation message 上会把主传播输入缩小为 50%

因此改成：

```text
selection_factor = 1 + strength * (selection_weight - 0.5)
obs_selected = obs_base * selection_factor
```

这样初始化时 `selection_factor = 1.0`，不会破坏主干。

| 配置 | 数据集 | 轮数 | strength | MSE 均值 ± std | max | 结论 |
|------|--------|------|----------|----------------|-----|------|
| `spikeselectprop_res010` | USHCN | 10 | `0.10` | `0.17535 ± 0.02540` | `0.24952` | 均值可接受，但仍有重坏轮 |
| `spikeselectprop_res005` | USHCN | 10 | `0.05` | `0.16988 ± 0.00937` | `0.19176` | 当前最优候选 |
| `spikeselectprop_res005` | HumanActivity | 5 | `0.05` | `0.04175 ± 0.00018` | `0.04202` | 稳定，不伤简单数据 |

新的约束：

1. 如果继续做 propagation selection，只允许 identity-preserving residual 形式。
2. 当前保留 `strength = 0.05`，不要继续放大到 `0.10`。
3. `HumanActivity` 已确认稳定，不伤简单数据。
4. 下一步安排服务器覆盖 `P12 / MIMIC_III`，暂不跑 `MIMIC_IV`。
