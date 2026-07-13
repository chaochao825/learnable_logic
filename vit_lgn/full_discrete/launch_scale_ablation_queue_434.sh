#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/wangmeiqi/learnable_logic_full_discrete_scale_ablation_20260713}
DATA_ROOT=${DATA_ROOT:-/home/wangmeiqi/ViT-LGN_goal6plus_nobias_retry_20260630/data/cifar-10}
PYTHON_BIN=${PYTHON_BIN:-python}
GPU_INDEX=${GPU_INDEX:-2}

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
if ! grep -Fq 'seed_all(args.seed + 30_000)' \
     "$ROOT/vit_lgn/full_discrete/train_cifar.py"; then
  echo "Scale protocol requires the post-construction data RNG reset" >&2
  exit 2
fi
if ! "$PYTHON_BIN" -c 'import torch, torchvision' >/dev/null 2>&1; then
  echo "PYTHON_BIN cannot import torch and torchvision" >&2
  exit 2
fi

WIDTH_LOCK=/tmp/codex_lgn_vit_fdscale_d6e384_seed42.lock
DEPTH_LOCK=/tmp/codex_lgn_vit_fdscale_d12e192_seed42.lock
CONTROL_LOCK=/tmp/codex_lgn_vit_fdscale_d6e192_control_seed42.lock
COMBINED_LOCK=/tmp/codex_lgn_vit_fdscale_d12e384_seed42.lock
GPU_LOCK=/tmp/codex_lgn_vit_50k_gpu${GPU_INDEX}.lock
exec 5>"$COMBINED_LOCK"
exec 6>"$WIDTH_LOCK"
exec 7>"$DEPTH_LOCK"
exec 8>"$CONTROL_LOCK"
if ! flock -n 5 || ! flock -n 6 || ! flock -n 7 || ! flock -n 8; then
  echo "A scale-ablation output is already running" >&2
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
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

run_variant() {
  local name=$1
  local dim=$2
  local depth=$3
  local heads=$4
  "$PYTHON_BIN" -m vit_lgn.full_discrete.train_cifar \
    --data-root "$DATA_ROOT" --out-dir "$ROOT/runs/$name" \
    --seed 42 --steps 50000 --batch-size 128 --eval-batch-size 256 \
    --dim "$dim" --depth "$depth" --heads "$heads" \
    --topk 8 --mlp-ratio 4 --weight-magnitude-bits 7 \
    --activation-bits 8 --qk-lanes 7 \
    --learning-rate 3e-4 --min-learning-rate 1e-5 --warmup-steps 2000 \
    --weight-decay 0.05 --eval-every 5000 --checkpoint-every 5000 \
    --resume >>"$ROOT/${name}.log" 2>&1
}

# Width-only, depth-only, same-host control, then the combined model.  The
# selected train_cifar resets the augmentation RNG after model construction,
# so all four runs share sampler order and crop/flip RNG streams.
run_variant fdscale_d6e384_seed42 384 6 6
run_variant fdscale_d12e192_seed42 192 12 6
run_variant fdscale_d6e192_control_seed42 192 6 6
run_variant fdscale_d12e384_seed42 384 12 6
