#!/usr/bin/env bash
set -euo pipefail

cd /home/spco/sow_linear/ViT-LGN_goal6plus

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONPATH="/home/wangmeiqi/pyf"
export PYTHONDONTWRITEBYTECODE=1

PY="/home/wangmeiqi/anaconda3/envs/convlogic/bin/python"
LOG_DIR="logs/20260628_cls_topk_token_mixer_ablation"
mkdir -p "${LOG_DIR}"
RUN_STAMP="$(date '+%Y%m%d_%H%M%S')_pid$$"
LOG_FILE="${LOG_DIR}/run_${RUN_STAMP}.log"

{
    echo "cls top-k token mixer ablation start $(date -Is)"
    echo "python=${PY}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
    "${PY}" cls_topk_token_mixer_ablation.py \
        --config-json runs/attn_best_depth_expand_pilot_l4_fixed/20260627_232055/config.json \
        --teacher-iters 1000 \
        --eval-split test \
        --eval-max-batches -1 \
        --route-stats-batches 20 \
        --block-settings "b3;b2,b3" \
        --topk-list 4,8,16 \
        --mixer-variants static_mean,static_weighted,dynamic_mean,dynamic_weighted \
        --out-dir runs/cls_topk_token_mixer_round1_topkmasked \
        --num-workers 2
    echo "cls top-k token mixer ablation done $(date -Is)"
} 2>&1 | tee "${LOG_FILE}"
