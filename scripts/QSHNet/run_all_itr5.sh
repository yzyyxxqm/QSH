#!/bin/bash
# =============================================================================
# QSH-Net 全数据集 itr=5 评估脚本
# 依次运行 USHCN, HumanActivity, P12, MIMIC_III，最后汇总所有结果
# 用法: bash scripts/QSHNet/run_all_itr5.sh
# =============================================================================

set -e  # 任何命令失败则停止

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

LOG_DIR="$PROJECT_DIR/storage/logs/QSHNet_run_all_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "QSH-Net 全数据集评估 (itr=5)"
echo "日志目录: $LOG_DIR"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ---------------------------------------------------------------------------
# 公共设置
# ---------------------------------------------------------------------------
. "$SCRIPT_DIR/../globals.sh"

use_multi_gpu=0
if [ $use_multi_gpu -eq 0 ]; then
    launch_command="python"
else
    launch_command="accelerate launch"
fi

ITR=5

run_dataset() {
    local dataset_name=$1
    local seq_len=$2
    local pred_len=$3
    local d_model=$4
    local n_layers=$5
    local n_heads=$6
    local batch_size=$7

    local dataset_subset_name=""
    local dataset_id=$dataset_name
    get_dataset_info "$dataset_name" "$dataset_subset_name"

    local model_name="QSHNet"
    local model_id="QSHNet"
    local log_file="$LOG_DIR/${dataset_name}.log"

    echo ""
    echo "============================================================"
    echo "[$(date '+%H:%M:%S')] 开始训练: $dataset_name"
    echo "  参数: seq=$seq_len pred=$pred_len d=$d_model layers=$n_layers heads=$n_heads bs=$batch_size itr=$ITR"
    echo "  日志: $log_file"
    echo "============================================================"

    $launch_command main.py \
        --is_training 1 \
        --collate_fn "collate_fn" \
        --loss "MSE" \
        --d_model $d_model \
        --n_layers $n_layers \
        --n_heads $n_heads \
        --use_multi_gpu $use_multi_gpu \
        --dataset_root_path $dataset_root_path \
        --model_id $model_id \
        --model_name $model_name \
        --dataset_name $dataset_name \
        --dataset_id $dataset_id \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in $n_variables \
        --dec_in $n_variables \
        --c_out $n_variables \
        --train_epochs 300 \
        --patience 10 \
        --val_interval 1 \
        --itr $ITR \
        --batch_size $batch_size \
        --learning_rate 1e-3 \
        2>&1 | tee "$log_file"

    echo "[$(date '+%H:%M:%S')] 完成: $dataset_name"
}

# ---------------------------------------------------------------------------
# 依次运行 4 个数据集
# ---------------------------------------------------------------------------
#              dataset_name   seq  pred  d_model  layers  heads  bs
run_dataset    "USHCN"        150  3     256      1       1      16
run_dataset    "HumanActivity" 3000 300  128      3       1      32
run_dataset    "P12"          36   3     256      2       8      32
run_dataset    "MIMIC_III"    72   3     256      2       4      32

# ---------------------------------------------------------------------------
# 汇总所有结果
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "所有数据集训练完成，正在汇总结果..."
echo "============================================================"

python3 - "$LOG_DIR" <<'PYTHON_SCRIPT'
import sys, re, json
import numpy as np
from pathlib import Path

log_dir = Path(sys.argv[1])

# HyperIMTS paper baselines
baselines = {
    "USHCN":         {"MSE": 0.1738, "std": 0.0078},
    "HumanActivity": {"MSE": 0.0421, "std": 0.0021},
    "P12":           {"MSE": 0.2996, "std": 0.0003},
    "MIMIC_III":     {"MSE": None,   "std": None},  # 论文未单独报告
}

datasets = ["USHCN", "HumanActivity", "P12", "MIMIC_III"]

# Parse metrics from log files
mse_pattern = re.compile(r'"MSE":\s*([\d.]+)')
mae_pattern = re.compile(r'"MAE":\s*([\d.]+)')

print()
print("=" * 80)
print("QSH-Net 全数据集评估结果汇总")
print("=" * 80)

summary = {}

for ds in datasets:
    log_file = log_dir / f"{ds}.log"
    if not log_file.exists():
        print(f"\n[{ds}] ⚠ 日志文件不存在: {log_file}")
        continue

    content = log_file.read_text()
    mse_values = [float(m) for m in mse_pattern.findall(content)]
    mae_values = [float(m) for m in mae_pattern.findall(content)]

    if not mse_values:
        print(f"\n[{ds}] ⚠ 未找到 MSE 结果")
        continue

    mse_arr = np.array(mse_values)
    mae_arr = np.array(mae_values) if mae_values else np.array([0.0])
    mean_mse = mse_arr.mean()
    std_mse = mse_arr.std()
    mean_mae = mae_arr.mean()
    std_mae = mae_arr.std()
    summary[ds] = {
        "mse_values": mse_values, "mse_mean": mean_mse, "mse_std": std_mse,
        "mae_values": mae_values, "mae_mean": mean_mae, "mae_std": std_mae,
    }

    print(f"\n{'─' * 60}")
    print(f"📊 {ds} (itr={len(mse_values)})")
    print(f"{'─' * 60}")
    for i, (mse_v, mae_v) in enumerate(zip(mse_values, mae_values)):
        print(f"  iter {i}: MSE = {mse_v:.6f}  MAE = {mae_v:.6f}")
    print(f"  ────────────────────────")
    print(f"  MSE 均值: {mean_mse:.6f} ± {std_mse:.6f}")
    print(f"  MAE 均值: {mean_mae:.6f} ± {std_mae:.6f}")

    bl = baselines.get(ds)
    if bl and bl["MSE"] is not None:
        delta = (mean_mse - bl["MSE"]) / bl["MSE"] * 100
        status = "✅ 超越" if mean_mse < bl["MSE"] else ("⚠ 持平" if abs(delta) < 1.0 else "❌ 未达")
        print(f"  HyperIMTS 论文: {bl['MSE']:.4f} ± {bl['std']:.4f}")
        print(f"  相对改善: {delta:+.2f}%  {status}")

# Final comparison table
print(f"\n\n{'=' * 80}")
print("对比总表")
print(f"{'=' * 80}")
print(f"{'数据集':<16} {'QSH-Net MSE':>18} {'HyperIMTS 论文':>18} {'改善':>10} {'状态':>6}")
print(f"{'─' * 80}")
for ds in datasets:
    if ds not in summary:
        print(f"{ds:<16} {'N/A':>18} ", end="")
    else:
        s = summary[ds]
        qsh_str = f"{s['mse_mean']:.4f}±{s['mse_std']:.4f}"
        print(f"{ds:<16} {qsh_str:>18} ", end="")

    bl = baselines.get(ds)
    if bl and bl["MSE"] is not None:
        bl_str = f"{bl['MSE']:.4f}±{bl['std']:.4f}"
        print(f"{bl_str:>18} ", end="")
        if ds in summary:
            delta = (summary[ds]["mse_mean"] - bl["MSE"]) / bl["MSE"] * 100
            status = "✅" if summary[ds]["mse_mean"] < bl["MSE"] else "❌"
            print(f"{delta:>+9.2f}% {status:>4}")
        else:
            print(f"{'N/A':>10} {'':>6}")
    else:
        print(f"{'N/A':>18} {'N/A':>10} {'':>6}")

print(f"{'─' * 80}")
print()

# Save summary to JSON
summary_file = log_dir / "summary.json"
json_summary = {}
for ds in datasets:
    if ds in summary:
        s = summary[ds]
        json_summary[ds] = {
            "mse_values": s["mse_values"],
            "mse_mean": round(float(s["mse_mean"]), 6),
            "mse_std": round(float(s["mse_std"]), 6),
            "mae_values": s["mae_values"],
            "mae_mean": round(float(s["mae_mean"]), 6),
            "mae_std": round(float(s["mae_std"]), 6),
        }
        bl = baselines.get(ds)
        if bl and bl["MSE"] is not None:
            json_summary[ds]["hyperimts_paper_mse"] = bl["MSE"]
            json_summary[ds]["relative_improvement_pct"] = round(
                (s["mse_mean"] - bl["MSE"]) / bl["MSE"] * 100, 2
            )

with open(summary_file, "w") as f:
    json.dump(json_summary, f, indent=2)
print(f"结果已保存至: {summary_file}")
PYTHON_SCRIPT

echo ""
echo "============================================================"
echo "全部完成！结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "日志目录: $LOG_DIR"
echo "============================================================"
