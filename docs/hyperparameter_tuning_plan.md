# QSH-Net 超参数调优方案

## 当前状态（QDAS 配置，itr=5）

| 数据集 | 均值 MSE | 标准差 | 目标 MSE | 差距 |
|--------|---------|--------|---------|------|
| P12 | 0.2997 | 0.0014 | <0.295 | +1.6% |
| HumanActivity | 0.0417 | 0.0002 | <0.04 | +4.3% |
| USHCN | 0.1950 | 0.0434 | <0.16 | +21.9% |

## 当前超参数（与 HyperIMTS 完全一致）

| 参数 | P12 | HumanActivity | USHCN |
|------|-----|---------------|-------|
| d_model | 256 | 128 | 256 |
| n_layers | 1 | 3 | 1 |
| n_heads | 8 | 1 | 1 |
| batch_size | 32 | 32 | 16 |
| learning_rate | 1e-3 | 1e-3 | 1e-3 |
| lr_scheduler | DelayedStepDecayLR | DelayedStepDecayLR | DelayedStepDecayLR |
| train_epochs | 300 | 300 | 300 |
| patience | 10 | 10 | 10 |
| optimizer | Adam | Adam | Adam |

## 问题诊断

- **P12**：稳定但系统性偏高 1.6%。标准差极小，说明不是随机问题。
- **HumanActivity**：稳定但系统性偏高 4.3%。与 E 基线完全一致，说明 QDAS 组件对此数据集无贡献。
- **USHCN**：方差极大（0.0434）。5 轮中 2 轮达标（0.165），3 轮未达标。问题是训练不稳定。

## 调优策略

### 第一轮：Learning Rate 调优（优先级最高）

当前 lr=1e-3 对所有数据集统一使用。QDAS 比 E 多了四元数解码器和自适应脉冲的参数，可能需要不同的 lr。

**实验矩阵：**

| 实验编号 | 数据集 | learning_rate | 其他参数 | itr |
|---------|--------|--------------|---------|-----|
| LR1 | USHCN | 5e-4 | 不变 | 5 |
| LR2 | USHCN | 2e-3 | 不变 | 5 |
| LR3 | HumanActivity | 5e-4 | 不变 | 3 |
| LR4 | HumanActivity | 2e-3 | 不变 | 3 |
| LR5 | P12 | 5e-4 | 不变 | 3 |
| LR6 | P12 | 2e-3 | 不变 | 3 |

**预期：**
- USHCN 用 5e-4 可能降低方差（更稳定的优化），均值可能改善
- HumanActivity 和 P12 的最优 lr 可能不是 1e-3

### 第二轮：LR Scheduler 调优（如果第一轮不够）

当前用 DelayedStepDecayLR（前 2 epoch 不变，之后每 epoch ×0.8）。衰减可能太快。

**实验：**

| 实验编号 | 数据集 | lr_scheduler | 备注 |
|---------|--------|-------------|------|
| SC1 | 全部 | CosineAnnealingLR | 更平滑的衰减，常用于 Transformer |
| SC2 | 全部 | ExponentialDecayLR | 每 epoch ×0.5，更激进 |

### 第三轮：USHCN 专项（如果前两轮不够）

USHCN 的不稳定性可能需要更强的正则化：

| 实验编号 | 参数 | 值 | 备注 |
|---------|------|---|------|
| US1 | d_model | 128 | 减小模型容量，降低过拟合风险 |
| US2 | batch_size | 8 | 更小的 batch 增加梯度噪声，可能帮助逃离局部最优 |
| US3 | patience | 20 | 更长的耐心，让模型训练更充分 |

## 运行方式

使用消融脚本的 QDAS 配置，手动覆盖超参数。示例：

```bash
# LR1: USHCN lr=5e-4
python main.py --is_training 1 --collate_fn collate_fn --loss MSE \
    --d_model 256 --n_layers 1 --n_heads 1 \
    --dataset_root_path storage/datasets/USHCN \
    --model_id "QSHNet_noQB_noQH_noSP_noCM_noQV_noQE" \
    --model_name QSHNet --dataset_name USHCN --dataset_id USHCN \
    --features M --seq_len 150 --pred_len 3 \
    --enc_in 5 --dec_in 5 --c_out 5 \
    --train_epochs 300 --patience 10 --val_interval 1 \
    --itr 5 --batch_size 16 --learning_rate 5e-4
```

## 成功标准

- P12 均值 MSE < 0.295
- HumanActivity 均值 MSE < 0.04
- USHCN 均值 MSE < 0.16

## 备注

- 每轮调优只改一个参数，保持其他不变
- 优先跑 USHCN（差距最大，方差最大，改善空间最大）
- 如果 lr 调优后 USHCN 均值仍 >0.18，考虑第三轮的模型容量调整
