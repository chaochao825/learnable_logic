#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/wangmeiqi/learnable_logic_full_discrete_d12_20260713}
DATA_ROOT=${DATA_ROOT:-/home/wangmeiqi/ViT-LGN_goal6plus_nobias_retry_20260630/data/cifar-10}
PYTHON_BIN=${PYTHON_BIN:-python}
GPU_INDEX=${GPU_INDEX:-0}

if [[ ! "$GPU_INDEX" =~ ^(0|[1-9][0-9]*)$ ]] || \
   ! nvidia-smi --id="$GPU_INDEX" --query-gpu=index --format=csv,noheader,nounits >/dev/null 2>&1; then
  echo "Invalid GPU_INDEX=$GPU_INDEX" >&2
  exit 2
fi
if ! git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 || \
   [[ ! -d "$DATA_ROOT/cifar-10-batches-py" ]]; then
  echo "Missing repository or CIFAR-10 data" >&2
  exit 2
fi

OUTPUT_LOCK=/tmp/codex_lgn_vit_full_discrete_d12e384_seed42.lock
GPU_LOCK=/tmp/codex_lgn_vit_50k_gpu${GPU_INDEX}.lock
exec 8>"$OUTPUT_LOCK"
if ! flock -n 8; then
  echo "full_discrete_d12e384_seed42 is already running" >&2
  exit 3
fi
exec 9>"$GPU_LOCK"
flock 9

while true; do
  read -r used util < <(nvidia-smi --id="$GPU_INDEX" \
    --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits \
    | tr -d ' ' | tr ',' ' ')
  (( used <= 1024 && util <= 10 )) && break
  sleep 60
done

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

"$PYTHON_BIN" -m vit_lgn.full_discrete.train_cifar \
  --data-root "$DATA_ROOT" \
  --out-dir "$ROOT/runs/full_discrete_d12e384_seed42" \
  --seed 42 --steps 50000 --batch-size 128 --eval-batch-size 256 \
  --dim 384 --depth 12 --heads 6 --topk 8 --mlp-ratio 4 \
  --weight-magnitude-bits 7 --activation-bits 8 --qk-lanes 7 \
  --learning-rate 3e-4 --min-learning-rate 1e-5 --warmup-steps 2000 \
  --weight-decay 0.05 --eval-every 5000 --checkpoint-every 5000 \
  --resume >>"$ROOT/full_discrete_d12e384_seed42.log" 2>&1
