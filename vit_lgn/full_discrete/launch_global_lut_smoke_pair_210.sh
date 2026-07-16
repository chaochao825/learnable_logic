#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/spco/sow_linear/learnable_logic_global_lut_20260716}
DATA_ROOT=${DATA_ROOT:-/home/spco/sow_linear/ViT-LGN_attention_clean_20260629/data/cifar-10}
PYTHON_BIN=${PYTHON_BIN:-python}
GPU_INDEX=${GPU_INDEX:-2}
GPU_POLL_SECONDS=${GPU_POLL_SECONDS:-60}
GPU_IDLE_SAMPLES=${GPU_IDLE_SAMPLES:-3}
EXPECTED_SOURCE_SET_SHA256=71fec7ad6ad78acad28e456b143151c47e84aff59c3ff22f032f6010a8df3b0b

SOURCE_FILES=(
  model.py
  enhanced_model.py
  enhancements_lut.py
  enhancements_spatial.py
  enhancements_logic_tree.py
  enhancements_hadamard.py
  enhancements_global_lut.py
  enhancements_expert.py
  __init__.py
  logic_backend.py
  shiftadd.py
  train_cifar.py
)

source_set_sha256() {
  local file
  for file in "${SOURCE_FILES[@]}"; do
    sha256sum "$ROOT/vit_lgn/full_discrete/$file" | cut -d' ' -f1
  done | sha256sum | cut -d' ' -f1
}

if [[ ! "$GPU_INDEX" =~ ^(0|[1-9][0-9]*)$ ]] || \
   ! nvidia-smi --id="$GPU_INDEX" --query-gpu=index \
     --format=csv,noheader,nounits >/dev/null 2>&1; then
  echo "Invalid GPU_INDEX=$GPU_INDEX" >&2
  exit 2
fi
if [[ ! -d "$DATA_ROOT/cifar-10-batches-py" ]]; then
  echo "CIFAR-10 payload is missing" >&2
  exit 2
fi
observed_source=$(source_set_sha256)
if [[ "$observed_source" != "$EXPECTED_SOURCE_SET_SHA256" ]]; then
  echo "Frozen source set mismatch: $observed_source" >&2
  exit 2
fi

queue_exit="$ROOT/global_lut_smoke_pair.exit"
printf '%s\n' running >"$queue_exit"
trap 'status=$?; printf "%s\n" "$status" >"$queue_exit"' EXIT

exec 8>"/tmp/codex_global_lut_smoke_pair_gpu${GPU_INDEX}.lock"
if ! flock -n 8; then
  echo "Global LUT smoke pair is already queued or running" >&2
  exit 3
fi

# Serialize with every other LGN-ViT queue on this GPU.  The pair-specific
# lock above prevents duplicate submissions; this shared lock prevents a
# TOCTOU launch race with the existing 50k queues.
exec 9>"/tmp/codex_lgn_vit_50k_gpu${GPU_INDEX}.lock"
flock 9

if [[ ! "$GPU_POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || \
   [[ ! "$GPU_IDLE_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPU_POLL_SECONDS and GPU_IDLE_SAMPLES must be positive integers" >&2
  exit 2
fi

idle_samples=0
while (( idle_samples < GPU_IDLE_SAMPLES )); do
  read -r used util < <(
    nvidia-smi --id="$GPU_INDEX" \
      --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits \
      | tr -d ' ' | tr ',' ' '
  )
  if (( used <= 1024 && util <= 10 )); then
    ((idle_samples += 1))
    echo "GPU$GPU_INDEX idle sample $idle_samples/$GPU_IDLE_SAMPLES: used=$used MiB util=$util%"
  else
    idle_samples=0
    echo "Waiting for GPU$GPU_INDEX: used=$used MiB util=$util%"
  fi
  if (( idle_samples < GPU_IDLE_SAMPLES )); then
    sleep "$GPU_POLL_SECONDS"
  fi
done

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate att
if ! "$PYTHON_BIN" -c 'import torch, torchvision' >/dev/null 2>&1; then
  echo "PYTHON_BIN cannot import torch and torchvision" >&2
  exit 2
fi

cd "$ROOT"
export PYTHONPATH=.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
printf '%s\n' "$EXPECTED_SOURCE_SET_SHA256" >global_lut_smoke_source_set_sha256.txt

names=(
  glut_attention_local6_d6e192_seed42_1k
  glut_parallel_tree_local6_d6e192_seed42_1k
)
mixers=(attention parallel_lut_tree)

for index in "${!names[@]}"; do
  if [[ $(source_set_sha256) != "$EXPECTED_SOURCE_SET_SHA256" ]]; then
    echo "Source changed while queue was running" >&2
    exit 4
  fi
  "$PYTHON_BIN" -m vit_lgn.full_discrete.train_cifar \
    --data-root "$DATA_ROOT" --out-dir "$ROOT/runs/${names[$index]}" \
    --seed 42 --steps 1000 --batch-size 128 --eval-batch-size 256 \
    --dim 192 --depth 6 --heads 6 --topk 8 --mlp-ratio 4 \
    --weight-magnitude-bits 7 --activation-bits 8 --qk-lanes 7 \
    --norm-kind rms_lut --final-norm-kind same \
    --global-mixer "${mixers[$index]}" \
    --global-lut-group-size 32 --global-lut-branch-shift 2 \
    --local-layers 6 --local-operator depthwise_shiftadd \
    --learning-rate 3e-4 --min-learning-rate 1e-5 --warmup-steps 200 \
    --weight-decay 0.05 --eval-every 500 --checkpoint-every 1000 \
    --resume >>"$ROOT/${names[$index]}.log" 2>&1
done
