#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/spco/sow_linear/learnable_logic_full_discrete_20260713
GPU_INDEX=${GPU_INDEX:-0}
if [[ ! "$GPU_INDEX" =~ ^[0-9]+$ ]] || \
   ! nvidia-smi --id="$GPU_INDEX" --query-gpu=index --format=csv,noheader,nounits >/dev/null 2>&1; then
  echo "Invalid GPU_INDEX=$GPU_INDEX" >&2
  exit 2
fi
OUTPUT_LOCK=/tmp/codex_lgn_vit_full_discrete_d12e384_smoke.lock
exec 8>"$OUTPUT_LOCK"
if ! flock -n 8; then
  echo "full_discrete_d12e384_smoke is already running" >&2
  exit 3
fi
LOCK=/tmp/codex_lgn_vit_50k_gpu${GPU_INDEX}.lock
exec 9>"$LOCK"
flock 9
while true; do
  read -r used util < <(nvidia-smi --id="$GPU_INDEX" --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')
  (( used <= 1024 && util <= 10 )) && break
  sleep 60
done

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate att
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python -m vit_lgn.full_discrete.train_cifar \
  --data-root /home/spco/sow_linear/ViT-LGN_attention_clean_20260629/data/cifar-10 \
  --out-dir "$ROOT/runs/full_discrete_d12e384_smoke" \
  --seed 1212 --steps 4 --warmup-steps 1 --eval-every 4 --checkpoint-every 4 \
  --batch-size 16 --eval-batch-size 128 --dim 384 --depth 12 --heads 6 \
  --topk 8 --mlp-ratio 4 --weight-magnitude-bits 7 --activation-bits 8 \
  --qk-lanes 7 --resume
