#!/bin/bash
# QSH-Net Hyperparameter Tuning Script
#
# Model: QDAS (Quaternion Decoder + Adaptive Spike)
# model_id: QSHNet_noQB_noQH_noSP_noCM_noQV_noQE
#
# Usage:
#   bash scripts/QSHNet/hp_tuning.sh LR1          # run single experiment
#   bash scripts/QSHNet/hp_tuning.sh ROUND1        # run all Round 1 experiments
#   bash scripts/QSHNet/hp_tuning.sh ALL            # run everything

. "$(dirname "$(readlink -f "$0")")/../globals.sh"

MODEL_NAME="QSHNet"
MODEL_ID="QSHNet_noQB_noQH_noSP_noCM_noQV_noQE"
FILTER="${1:-ALL}"

run_exp() {
    local exp_id="$1"
    local dataset_name="$2"
    local lr="$3"
    local lr_sched="$4"
    local d_model="$5"
    local batch_size="$6"
    local patience="$7"
    local itr="$8"

    if [ "$FILTER" != "ALL" ] && [ "$FILTER" != "$exp_id" ] && [ "$FILTER" != "ROUND1" ] && [ "$FILTER" != "ROUND2" ] && [ "$FILTER" != "ROUND3" ]; then
        return
    fi
    # Round filtering
    case "$exp_id" in
        LR*) [ "$FILTER" = "ROUND2" ] || [ "$FILTER" = "ROUND3" ] && return ;;
        SC*) [ "$FILTER" = "ROUND1" ] || [ "$FILTER" = "ROUND3" ] && return ;;
        US*) [ "$FILTER" = "ROUND1" ] || [ "$FILTER" = "ROUND2" ] && return ;;
    esac

    get_dataset_info "$dataset_name" ""

    local n_layers=2; local n_heads=4; local seq_len=96; local pred_len=3
    local extra_args=""
    case $dataset_name in
        P12)       d_model=${d_model:-256}; n_layers=1; n_heads=8; seq_len=36; pred_len=3; batch_size=${batch_size:-32}; extra_args="--collate_fn collate_fn" ;;
        HumanActivity) d_model=${d_model:-128}; n_layers=3; n_heads=1; seq_len=3000; pred_len=300; batch_size=${batch_size:-32}; extra_args="--collate_fn collate_fn" ;;
        USHCN)     d_model=${d_model:-256}; n_layers=1; n_heads=1; seq_len=150; pred_len=3; batch_size=${batch_size:-16}; extra_args="--collate_fn collate_fn" ;;
    esac

    echo ""
    echo "============================================"
    echo "  [$exp_id] $dataset_name lr=$lr sched=$lr_sched d=$d_model bs=$batch_size pat=$patience itr=$itr"
    echo "============================================"
    python main.py \
        --is_training 1 --loss MSE \
        --d_model $d_model --n_layers $n_layers --n_heads $n_heads \
        --use_multi_gpu 0 \
        --dataset_root_path $dataset_root_path \
        --model_id "${MODEL_ID}_${exp_id}" --model_name $MODEL_NAME \
        --dataset_name $dataset_name --dataset_id $dataset_name \
        --features M --seq_len $seq_len --pred_len $pred_len \
        --enc_in $n_variables --dec_in $n_variables --c_out $n_variables \
        --train_epochs 300 --patience $patience --val_interval 1 \
        --itr $itr --batch_size $batch_size \
        --learning_rate $lr --lr_scheduler $lr_sched \
        $extra_args
}

# ============================================================
# Round 1: Learning Rate (priority: USHCN > Human > P12)
# ============================================================
# USHCN: biggest gap, highest variance — try lower lr for stability
run_exp "LR1" "USHCN"         5e-4 "DelayedStepDecayLR" "" "" 10 5
run_exp "LR2" "USHCN"         2e-3 "DelayedStepDecayLR" "" "" 10 5
# HumanActivity: stable but 4.3% gap
run_exp "LR3" "HumanActivity" 5e-4 "DelayedStepDecayLR" "" "" 10 3
run_exp "LR4" "HumanActivity" 2e-3 "DelayedStepDecayLR" "" "" 10 3
# P12: stable but 1.6% gap
run_exp "LR5" "P12"           5e-4 "DelayedStepDecayLR" "" "" 10 3
run_exp "LR6" "P12"           2e-3 "DelayedStepDecayLR" "" "" 10 3

# ============================================================
# Round 2: LR Scheduler (use best lr from Round 1, or 1e-3 default)
# ============================================================
run_exp "SC1" "USHCN"         1e-3 "CosineAnnealingLR"  "" "" 10 5
run_exp "SC2" "HumanActivity" 1e-3 "CosineAnnealingLR"  "" "" 10 3
run_exp "SC3" "P12"           1e-3 "CosineAnnealingLR"  "" "" 10 3

# ============================================================
# Round 3: USHCN-specific (model capacity + training config)
# ============================================================
run_exp "US1" "USHCN"         5e-4 "DelayedStepDecayLR" 128 "" 10 5
run_exp "US2" "USHCN"         5e-4 "DelayedStepDecayLR" "" 8  10 5
run_exp "US3" "USHCN"         1e-3 "DelayedStepDecayLR" "" "" 20 5

echo ""
echo "=== Tuning complete ==="
