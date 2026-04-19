# QSH-Net Coupled Residual Bound 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在 `apply_event_injection(...)` 中新增“耦合后残差相对主状态的软上界”，用于压制 `USHCN` 坏轮，同时尽量不伤 `HumanActivity / P12 / MIMIC_III`。

**架构：** 保持当前 `eventscalecap / eventdensvar` 的 `event` 注入、双支路归一化、density modulation、quaternion residual 全部不变，只在 `main_state + event_scale * event_delta` 这一步增加一个新的 `coupled_residual_ratio_max` 约束。通过新增单元测试锁定其行为，并补充最小诊断字段，便于后续看 `clip_rate` 是否只在高风险 batch 上触发。

**技术栈：** Python, PyTorch, unittest, PyOmniTS/QSHNet

---

## 文件结构

**修改：**
- `models/QSHNet.py`
  责任：新增 `coupled_residual_ratio_max`、实现耦合残差 soft ratio cap、输出最小诊断统计。
- `tests/models/test_QSHNet.py`
  责任：新增针对耦合残差约束的失败测试与通过测试，锁定行为。
- `exp/exp_main.py`
  责任：把新的诊断字段纳入 `QSHDiag`，便于后续本地筛查。

**参考：**
- `docs/superpowers/specs/2026-04-18-qshnet-coupled-residual-bound-design.md`

### 任务 1：为耦合残差上界补测试

**文件：**
- 修改：`tests/models/test_QSHNet.py`
- 参考：`models/QSHNet.py`

- [ ] **步骤 1：编写失败的测试，锁定基线初始化和上界行为**

```python
    def test_coupled_residual_ratio_max_initializes_to_wide_safe_default(self):
        learner = HypergraphLearner(n_layers=2, d_model=8, n_heads=1, time_length=4)

        self.assertAlmostEqual(learner.coupled_residual_ratio_max, 0.20)

    def test_coupled_residual_bound_keeps_small_event_injection_unchanged(self):
        learner = HypergraphLearner(n_layers=1, d_model=8, n_heads=1, time_length=4)
        main_state = torch.ones(2, 3, 8)
        event_delta = torch.full((2, 3, 8), 0.1)
        event_scale = torch.tensor(0.1)

        bounded_state, diag = learner.apply_event_injection(
            layer_idx=0,
            main_state=main_state,
            event_delta=event_delta,
            event_scale=event_scale,
            target="temporal",
            return_diagnostics=True,
        )

        expected = main_state + 0.01
        self.assertTrue(torch.allclose(bounded_state, expected, atol=1e-6))
        self.assertEqual(diag["clip_rate"], 0.0)

    def test_coupled_residual_bound_scales_down_large_event_injection(self):
        learner = HypergraphLearner(n_layers=1, d_model=8, n_heads=1, time_length=4)
        main_state = torch.ones(2, 3, 8)
        event_delta = torch.full((2, 3, 8), 10.0)
        event_scale = torch.tensor(0.1)

        bounded_state, diag = learner.apply_event_injection(
            layer_idx=0,
            main_state=main_state,
            event_delta=event_delta,
            event_scale=event_scale,
            target="variable",
            return_diagnostics=True,
        )

        coupled_residual = bounded_state - main_state
        residual_norm = coupled_residual.norm(dim=-1)
        main_norm = main_state.norm(dim=-1)

        self.assertTrue(torch.all(residual_norm <= 0.20 * main_norm + 1e-6))
        self.assertGreater(diag["clip_rate"], 0.0)
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
python -m unittest tests.models.test_QSHNet.TestQSHNet.test_coupled_residual_ratio_max_initializes_to_wide_safe_default tests.models.test_QSHNet.TestQSHNet.test_coupled_residual_bound_keeps_small_event_injection_unchanged tests.models.test_QSHNet.TestQSHNet.test_coupled_residual_bound_scales_down_large_event_injection
```

预期：

- FAIL
- 报错 `HypergraphLearner` 缺少 `coupled_residual_ratio_max`
- 或 `apply_event_injection()` 不接受 `return_diagnostics`

- [ ] **步骤 3：补充现有测试，兼容新的返回格式**

把现有的 `test_event_injection_adds_bounded_delta_to_both_hyperedge_paths` 改为：

```python
        temporal_injected, temporal_diag = learner.apply_event_injection(
            layer_idx=0,
            main_state=main_state,
            event_delta=event_delta,
            event_scale=event_scale,
            target="temporal",
            return_diagnostics=True,
        )
        variable_injected, variable_diag = learner.apply_event_injection(
            layer_idx=0,
            main_state=main_state,
            event_delta=event_delta,
            event_scale=event_scale,
            target="variable",
            return_diagnostics=True,
        )

        expected = main_state + 0.05
        self.assertTrue(torch.allclose(temporal_injected, expected, atol=1e-6))
        self.assertTrue(torch.allclose(variable_injected, expected, atol=1e-6))
        self.assertEqual(temporal_diag["clip_rate"], 0.0)
        self.assertEqual(variable_diag["clip_rate"], 0.0)
```

- [ ] **步骤 4：Commit**

```bash
git add tests/models/test_QSHNet.py
git commit -m "test: add coupled residual bound coverage"
```

### 任务 2：在 `QSHNet.py` 中实现耦合残差 soft ratio cap

**文件：**
- 修改：`models/QSHNet.py`
- 测试：`tests/models/test_QSHNet.py`

- [ ] **步骤 1：新增配置字段和核心辅助函数**

在 `HypergraphLearner.__init__` 中新增：

```python
        self.coupled_residual_ratio_max = 0.20
```

在 `HypergraphLearner` 中新增辅助函数：

```python
    def bound_coupled_residual(self, main_state, coupled_residual):
        main_norm = main_state.norm(dim=-1, keepdim=True)
        residual_norm = coupled_residual.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        max_residual_norm = self.coupled_residual_ratio_max * main_norm
        residual_scale = torch.clamp(max_residual_norm / residual_norm, max=1.0)
        bounded_residual = coupled_residual * residual_scale
        clip_rate = (residual_scale < 1.0).float().mean().item()
        ratio_mean = (residual_norm / main_norm.clamp(min=1e-6)).mean().item()
        ratio_max = (residual_norm / main_norm.clamp(min=1e-6)).max().item()
        return bounded_residual, {
            "coupled_residual_norm_mean": bounded_residual.norm(dim=-1).mean().item(),
            "coupled_residual_ratio_mean": ratio_mean,
            "coupled_residual_ratio_max": ratio_max,
            "clip_rate": clip_rate,
        }
```

- [ ] **步骤 2：让 `apply_event_injection(...)` 返回有界状态和诊断**

把 `apply_event_injection(...)` 改成：

```python
    def apply_event_injection(
        self,
        layer_idx,
        main_state,
        event_delta,
        event_scale,
        target,
        route_density=None,
        return_diagnostics=False,
    ):
        if target not in {"temporal", "variable"}:
            raise ValueError(f"Unknown event target: {target}")
        if route_density is not None:
            event_scale = self.modulate_event_scale(event_scale, route_density, target)
        coupled_residual = event_scale * event_delta
        bounded_residual, diag = self.bound_coupled_residual(main_state, coupled_residual)
        bounded_state = main_state + bounded_residual
        if return_diagnostics:
            return bounded_state, diag
        return bounded_state
```

- [ ] **步骤 3：在前向里接入新返回值，并记录每层 event 注入诊断**

把两个注入调用改成：

```python
            temporal_hyperedges_updated, temporal_event_diag = self.apply_event_injection(
                layer_idx=i,
                main_state=temporal_hyperedges_updated,
                event_delta=temporal_event_delta,
                event_scale=event_scale,
                target="temporal",
                route_density=temporal_route_density,
                return_diagnostics=True,
            )
```

```python
            variable_hyperedges_updated, variable_event_diag = self.apply_event_injection(
                layer_idx=i,
                main_state=variable_hyperedges_updated,
                event_delta=variable_event_delta,
                event_scale=event_scale,
                target="variable",
                route_density=variable_route_density,
                return_diagnostics=True,
            )
```

并在现有 layer 诊断附近增加：

```python
            route_state["temporal_event_diag"] = temporal_event_diag
            route_state["variable_event_diag"] = variable_event_diag
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
python -m unittest tests.models.test_QSHNet -v
```

预期：

- PASS

- [ ] **步骤 5：运行语法检查**

运行：

```bash
python -m py_compile models/QSHNet.py tests/models/test_QSHNet.py
```

预期：

- 无输出

- [ ] **步骤 6：Commit**

```bash
git add models/QSHNet.py tests/models/test_QSHNet.py
git commit -m "feat: bound coupled event residual in qshnet"
```

### 任务 3：把新诊断接入 `QSHDiag`

**文件：**
- 修改：`exp/exp_main.py`
- 参考：`models/QSHNet.py`

- [ ] **步骤 1：在诊断收集中增加新的 event 注入统计**

在 `_log_qsh_diagnostics(...)` 中，沿用现有 `QSHDiag` 拼接风格，新增：

```python
                    if hasattr(layer, "temporal_event_diag"):
                        parts.append(
                            f"temporal_clip={layer.temporal_event_diag['clip_rate']:.4f}"
                        )
```

更实际的做法是从 `model.model` 或对应 `HypergraphLearner` 读取最近一次 forward 缓存的：

- `temporal_event_diag`
- `variable_event_diag`

并记录：

- `temporal_clip`
- `variable_clip`
- `temporal_ratio_mean`
- `variable_ratio_mean`

- [ ] **步骤 2：在 `QSHNet.py` 中缓存最近一次 layer 级诊断**

在前向里加：

```python
            if not hasattr(self, "latest_event_diagnostics"):
                self.latest_event_diagnostics = {}
            self.latest_event_diagnostics[i] = {
                "temporal": temporal_event_diag,
                "variable": variable_event_diag,
            }
```

- [ ] **步骤 3：运行最小验证**

运行：

```bash
python -m py_compile exp/exp_main.py models/QSHNet.py
python -m unittest tests.models.test_QSHNet -v
```

预期：

- 语法检查通过
- 单元测试仍然 PASS

- [ ] **步骤 4：Commit**

```bash
git add exp/exp_main.py models/QSHNet.py
git commit -m "feat: log coupled residual diagnostics"
```

### 任务 4：本地最小验证顺序

**文件：**
- 修改：无
- 运行：现有训练入口
- 参考：`docs/superpowers/specs/2026-04-18-qshnet-coupled-residual-bound-design.md`

- [ ] **步骤 1：HumanActivity 快速筛查**

运行：

```bash
python main.py --is_training 1 --collate_fn collate_fn --loss MSE \
    --d_model 128 --n_layers 3 --n_heads 1 \
    --dataset_root_path storage/datasets/HumanActivity \
    --model_id coupledres_main --model_name QSHNet \
    --dataset_name HumanActivity --dataset_id HumanActivity \
    --features M --seq_len 3000 --pred_len 300 \
    --enc_in 12 --dec_in 12 --c_out 12 \
    --train_epochs 300 --patience 10 --val_interval 1 \
    --itr 3 --batch_size 32 --learning_rate 1e-3
```

预期：

- 不明显差于当前 `eventdensvar_main`

- [ ] **步骤 2：USHCN 本地筛查**

运行：

```bash
python main.py --is_training 1 --collate_fn collate_fn --loss MSE \
    --d_model 256 --n_layers 1 --n_heads 1 \
    --dataset_root_path storage/datasets/USHCN \
    --model_id coupledres_main --model_name QSHNet \
    --dataset_name USHCN --dataset_id USHCN \
    --features M --seq_len 150 --pred_len 3 \
    --enc_in 5 --dec_in 5 --c_out 5 \
    --train_epochs 300 --patience 10 --val_interval 1 \
    --itr 5 --batch_size 16 --learning_rate 1e-3
```

预期：

- 不能明显差于当前 `eventdensvar_main`
- `QSHDiag` 能看到 `clip_rate` 在 `USHCN` 上被局部触发

- [ ] **步骤 3：USHCN 正式判断**

运行：

```bash
python main.py --is_training 1 --collate_fn collate_fn --loss MSE \
    --d_model 256 --n_layers 1 --n_heads 1 \
    --dataset_root_path storage/datasets/USHCN \
    --model_id coupledres_itr10 --model_name QSHNet \
    --dataset_name USHCN --dataset_id USHCN \
    --features M --seq_len 150 --pred_len 3 \
    --enc_in 5 --dec_in 5 --c_out 5 \
    --train_epochs 300 --patience 10 --val_interval 1 \
    --itr 10 --batch_size 16 --learning_rate 1e-3
```

预期成功门槛：

- `mean <= 0.178`
- `std <= 0.02`

- [ ] **步骤 4：若 `USHCN itr=10` 达标，再送 `P12 / MIMIC_III`**

运行：

```bash
bash scripts/QSHNet/server_validate_eventdensvar_all.sh
```

附加要求：

- 仅在脚本切到新 `model_id` 后执行
- `P12 / MIMIC_III` 不明显退化

- [ ] **步骤 5：Commit**

```bash
git add .
git commit -m "chore: validate coupled residual bound locally"
```
