#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/spco/sow_linear/learnable_logic_full_discrete_20260713
LOCK=/tmp/codex_lgn_vit_50k_gpu0.lock
LOG="$ROOT/full_discrete_w8a8_seed42.log"
EXIT_FILE="$ROOT/full_discrete_w8a8_seed42.exit"

exec 9>"$LOCK"
flock 9
while true; do
  read -r used util < <(nvidia-smi --id=0 --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')
  (( used <= 1024 && util <= 10 )) && break
  sleep 60
done

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate att
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# A real GPU forward/backward/checkpoint smoke uses the same implementation.
python -m vit_lgn.full_discrete.train_cifar \
  --data-root /home/spco/sow_linear/ViT-LGN_attention_clean_20260629/data/cifar-10 \
  --out-dir "$ROOT/runs/full_discrete_smoke" \
  --seed 4242 --steps 4 --warmup-steps 1 --eval-every 4 --checkpoint-every 4 \
  --batch-size 8 --eval-batch-size 256 --dim 96 --depth 2 --heads 3 \
  --weight-magnitude-bits 7 --activation-bits 8 --qk-lanes 7 \
  --resume

set +e
python -m vit_lgn.full_discrete.train_cifar \
  --data-root /home/spco/sow_linear/ViT-LGN_attention_clean_20260629/data/cifar-10 \
  --out-dir "$ROOT/runs/full_discrete_w8a8_d6e192_seed42" \
  --seed 42 --steps 50000 --batch-size 128 --eval-batch-size 256 \
  --dim 192 --depth 6 --heads 6 --topk 8 --mlp-ratio 4 \
  --weight-magnitude-bits 7 --activation-bits 8 --qk-lanes 7 \
  --learning-rate 3e-4 --min-learning-rate 1e-5 --warmup-steps 2000 \
  --weight-decay 0.05 --eval-every 5000 --checkpoint-every 5000 \
  --resume \
  >>"$LOG" 2>&1
status=$?
set -e
printf '%s\n' "$status" >"$EXIT_FILE"
exit "$status"
