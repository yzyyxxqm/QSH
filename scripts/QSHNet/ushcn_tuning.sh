#!/bin/bash
# USHCN Training Stability Tuning
# Goal: reduce variance and improve mean MSE on USHCN
#
# Usage:
#   bash scripts/QSHNet/ushcn_tuning.sh T1       # single experiment
#   bash scripts/QSHNet/ushcn_tuning.sh ALL       # all experiments

. "$(dirname "$(readlink -f "$0")")/../globals.sh"

FILTER="${1:-ALL}"
get_dataset_info "USHCN" ""

run_exp() {
    local exp_id="$1"
    local lr="$2"
    local bs="$3"
    local patience="$4"

    if [ "$FILTER" != "ALL" ] && [ "$FILTER" != "$exp_id" ]; then return; fi

    echo ""
    echo "============================================"
    echo "  [$exp_id] USHCN lr=$lr bs=$bs patience=$patience"
    echo "============================================"
    python main.py \
        --is_training 1 --loss MSE \
        --d_model 256 --n_layers 1 --n_heads 1 \
        --use_multi_gpu 0 \
        --dataset_root_path $dataset_root_path \
        --model_id "QSHNet_${exp_id}" --model_name QSHNet \
        --dataset_name USHCN --dataset_id USHCN \
        --features M --seq_len 150 --pred_len 3 \
        --enc_in $n_variables --dec_in $n_variables --c_out $n_variables \
        --train_epochs 300 --patience $patience --val_interval 1 \
        --itr 5 --batch_size $bs \
        --learning_rate $lr \
        --collate_fn collate_fn
}

# Baseline: current config (for reference)
run_exp "T0" 1e-3 16 10

# Lower learning rate (most likely to stabilize)
run_exp "T1" 5e-4 16 10
run_exp "T2" 3e-4 16 10

# Larger batch size (reduces gradient noise)
run_exp "T3" 1e-3 32 10

# Lower lr + larger batch
run_exp "T4" 5e-4 32 10

# More patience (let training converge further)
run_exp "T5" 5e-4 16 20

echo ""
echo "=== USHCN tuning complete ==="
