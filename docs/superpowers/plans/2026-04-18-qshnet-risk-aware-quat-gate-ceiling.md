# QSHNet Risk-Aware Quaternion Gate Ceiling 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 仅通过限制高风险状态下的 quaternion gate 有效增益，压低 `USHCN itr=10` 的坏轮上界。

**架构：** 在现有 `alpha_raw` 之上增加一个风险感知的缩放因子 `cap_scale`。风险信号直接复用 fused-risk 诊断，不改 event、retain、fused residual 主线结构。

**技术栈：** Python, PyTorch, unittest

---

### 任务 1：为风险感知 quaternion gate ceiling 编写失败测试

**文件：**
- 修改：`/opt/Codes/PyOmniTS/tests/models/test_QSHNet.py`
- 测试：`/opt/Codes/PyOmniTS/tests/models/test_QSHNet.py`

- [ ] **步骤 1：添加失败测试，描述高风险下 gate 会被压低、低风险下保持原值**

```python
    def test_quaternion_gate_ceiling_reduces_alpha_only_under_high_risk(self):
        learner = HypergraphLearner(n_layers=1, d_model=8, n_heads=1, time_length=4)
        alpha_raw = torch.full((2, 3, 1), 0.08)
        low_risk = torch.zeros(2, 3, 1)
        high_risk = torch.ones(2, 3, 1)

        low_alpha = learner.apply_quaternion_risk_ceiling(alpha_raw, low_risk)
        high_alpha = learner.apply_quaternion_risk_ceiling(alpha_raw, high_risk)

        self.assertTrue(torch.allclose(low_alpha, alpha_raw))
        self.assertTrue(torch.all(high_alpha < low_alpha))
```

- [ ] **步骤 2：运行单测，确认它因方法缺失而失败**

运行：`conda run -n pyomnits python -m unittest tests.models.test_QSHNet.TestQSHNet.test_quaternion_gate_ceiling_reduces_alpha_only_under_high_risk -v`

预期：FAIL，提示 `HypergraphLearner` 缺少 `apply_quaternion_risk_ceiling`

### 任务 2：实现最小生产代码

**文件：**
- 修改：`/opt/Codes/PyOmniTS/models/QSHNet.py`

- [ ] **步骤 1：新增风险感知 ceiling 参数与 helper**

```python
        self.quat_risk_ceiling_min = 0.65

    def apply_quaternion_risk_ceiling(self, alpha_raw, fused_risk_pressure):
        cap_scale = 1.0 - (1.0 - self.quat_risk_ceiling_min) * fused_risk_pressure
        return alpha_raw * cap_scale.clamp(min=self.quat_risk_ceiling_min, max=1.0)
```

- [ ] **步骤 2：在 quaternion gate 前向路径中接入该 helper**

```python
            alpha = self.compute_quaternion_gate(i, linear_out, route_state["event_gate"])
            fused_risk_pressure = 1.0 - fused_event_diag["adaptive_ratio_max_mean"] / self.coupled_residual_ratio_max
            alpha = self.apply_quaternion_risk_ceiling(
                alpha,
                torch.full_like(alpha, fused_risk_pressure),
            )
            quat_residual = self.bound_quaternion_residual(linear_out, quat_out, alpha)
```

### 任务 3：验证绿灯

**文件：**
- 测试：`/opt/Codes/PyOmniTS/tests/models/test_QSHNet.py`

- [ ] **步骤 1：重新运行新增单测**

运行：`conda run -n pyomnits python -m unittest tests.models.test_QSHNet.TestQSHNet.test_quaternion_gate_ceiling_reduces_alpha_only_under_high_risk -v`

预期：PASS

- [ ] **步骤 2：运行完整 QSHNet 单测**

运行：`conda run -n pyomnits python -m unittest tests.models.test_QSHNet -v`

预期：PASS

- [ ] **步骤 3：运行语法检查**

运行：`python -m py_compile models/QSHNet.py tests/models/test_QSHNet.py exp/exp_main.py`

预期：无输出，退出码 0

### 任务 4：运行结构验证实验

**文件：**
- 修改：无
- 运行：`/opt/Codes/PyOmniTS/main.py`

- [ ] **步骤 1：运行 `USHCN itr=10`**

运行：

```bash
conda run -n pyomnits python main.py --is_training 1 --collate_fn collate_fn --loss MSE --d_model 256 --n_layers 1 --n_heads 1 --dataset_root_path storage/datasets/USHCN --model_id coupledctxadapt_main --model_name QSHNet --dataset_name USHCN --dataset_id USHCN --features M --seq_len 150 --pred_len 3 --enc_in 5 --dec_in 5 --c_out 5 --train_epochs 300 --patience 10 --val_interval 1 --itr 10 --batch_size 16 --learning_rate 1e-3
```

预期：产生新的结果目录和 10 个 `metric.json`

- [ ] **步骤 2：汇总结果并检查坏轮上界**

运行：读取新目录下所有 `iter*/eval_*/metric.json`

预期：
- 首要目标：`max_mse <= 0.18`
- 次要目标：均值不显著劣化

