#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data2/wangmeiqi/learnable_logic_hadamard_mixer_20260716}
DATA_ROOT=${DATA_ROOT:-/home/wangmeiqi/ViT-LGN_goal6plus_nobias_retry_20260630/data/cifar-10}
PYTHON_BIN=${PYTHON_BIN:-/data2/wangmeiqi/anaconda3/envs/syr_vit_train/bin/python}
GPU_INDEX=${GPU_INDEX:-2}
LOCAL_LAYERS=${LOCAL_LAYERS:-4}
RUN_NAME=${RUN_NAME:-scalelogic_d12e384_h12_local${LOCAL_LAYERS}_seed42_50k}
EXPECTED_SOURCE_SET_SHA256=0b5c9a9ad714612c3e55dbaec67415c8fe18ac8cec3e5e72c8fd671fcff9ef96

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
if [[ ! "$LOCAL_LAYERS" =~ ^(0|4)$ ]]; then
  echo "LOCAL_LAYERS must be 0 or 4 for the frozen pair" >&2
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

exec 9>"/tmp/codex_scalelogic_gpu${GPU_INDEX}.lock"
if ! flock -n 9; then
  echo "ScaleLogic GPU lock is already held" >&2
  exit 3
fi
read -r used util < <(
  nvidia-smi --id="$GPU_INDEX" \
    --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits \
    | tr -d ' ' | tr ',' ' '
)
if (( used > 1024 || util > 10 )); then
  echo "GPU$GPU_INDEX is not idle: used=$used MiB util=$util%" >&2
  exit 3
fi

cd "$ROOT"
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

"$PYTHON_BIN" -m vit_lgn.full_discrete.train_cifar \
  --data-root "$DATA_ROOT" \
  --out-dir "$ROOT/runs/$RUN_NAME" \
  --seed 42 --steps 50000 --batch-size 128 --eval-batch-size 256 \
  --dim 384 --depth 12 --heads 12 --topk 8 --mlp-ratio 4 \
  --weight-magnitude-bits 7 --activation-bits 8 --qk-lanes 7 \
  --norm-kind rms_lut --final-norm-kind same \
  --global-mixer attention \
  --local-layers "$LOCAL_LAYERS" --local-operator depthwise_shiftadd \
  --learning-rate 3e-4 --min-learning-rate 1e-5 --warmup-steps 2000 \
  --weight-decay 0.05 --eval-every 5000 --checkpoint-every 5000 \
  --resume
