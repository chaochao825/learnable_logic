#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/spco/sow_linear/learnable_logic_logic_backend_20260713}
DATA_ROOT=${DATA_ROOT:-/home/spco/sow_linear/ViT-LGN_attention_clean_20260629/data/cifar-10}
PYTHON_BIN=${PYTHON_BIN:-python}
GPU_INDEX=${GPU_INDEX:-2}

if [[ ! "$GPU_INDEX" =~ ^(0|[1-9][0-9]*)$ ]] || \
   ! nvidia-smi --id="$GPU_INDEX" --query-gpu=index \
     --format=csv,noheader,nounits >/dev/null 2>&1; then
  echo "Invalid GPU_INDEX=$GPU_INDEX" >&2
  exit 2
fi
if ! git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 || \
   [[ ! -d "$DATA_ROOT/cifar-10-batches-py" ]]; then
  echo "Missing repository or CIFAR-10 data" >&2
  exit 2
fi
if ! grep -Fq '"logic_backend.py"' \
     "$ROOT/vit_lgn/full_discrete/train_cifar.py"; then
  echo "Protocol must hash logic_backend.py before formal training" >&2
  exit 2
fi

PINNED_COMMIT=$(git -C "$ROOT" rev-parse HEAD)
assert_source_tree_frozen() {
  if [[ "$(git -C "$ROOT" rev-parse HEAD)" != "$PINNED_COMMIT" ]] || \
     ! git -C "$ROOT" diff --quiet "$PINNED_COMMIT" -- vit_lgn/full_discrete || \
     [[ -n "$(git -C "$ROOT" ls-files --others --exclude-standard \
       vit_lgn/full_discrete)" ]]; then
    echo "Source tree changed from pinned commit $PINNED_COMMIT" >&2
    exit 4
  fi
}
assert_source_tree_frozen

names=(
  fdlogic_wmag7_rms_seed42
  fdlogic_wmag7_shift_rms_seed42
  fdlogic_wmag4_rms_seed42
  fdlogic_wmag7_requant_seed42
  fdlogic_wmag7_none_seed42
  fdlogic_wmag7_shift_finalnone_seed42
  fdlogic_wmag7_rms_finalnone_seed42
)

GPU_LOCK=/tmp/codex_lgn_vit_50k_gpu${GPU_INDEX}.lock
exec 9>"$GPU_LOCK"
flock 9
for index in "${!names[@]}"; do
  descriptor=$((20 + index))
  eval "exec ${descriptor}>/tmp/codex_lgn_vit_${names[$index]}.lock"
  if ! flock -n "$descriptor"; then
    echo "${names[$index]} is already running" >&2
    exit 3
  fi
done

while true; do
  read -r used util < <(nvidia-smi --id="$GPU_INDEX" \
    --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits \
    | tr -d ' ' | tr ',' ' ')
  (( used <= 1024 && util <= 10 )) && break
  sleep 60
done

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate att
if ! "$PYTHON_BIN" -c 'import torch, torchvision' >/dev/null 2>&1; then
  echo "PYTHON_BIN cannot import torch and torchvision" >&2
  exit 2
fi

cd "$ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
printf '%s\n' "$PINNED_COMMIT" >logic_norm_queue_commit.txt

queue_exit="$ROOT/logic_norm_queue.exit"
trap 'status=$?; printf "%s\n" "$status" >"$queue_exit"' EXIT

run_variant() {
  local name=$1
  local magnitude_bits=$2
  local block_norm=$3
  local final_norm=$4
  assert_source_tree_frozen
  "$PYTHON_BIN" -m vit_lgn.full_discrete.train_cifar \
    --data-root "$DATA_ROOT" --out-dir "$ROOT/runs/$name" \
    --seed 42 --steps 50000 --batch-size 128 --eval-batch-size 256 \
    --dim 192 --depth 6 --heads 6 --topk 8 --mlp-ratio 4 \
    --weight-magnitude-bits "$magnitude_bits" --activation-bits 8 --qk-lanes 7 \
    --norm-kind "$block_norm" --final-norm-kind "$final_norm" \
    --learning-rate 3e-4 --min-learning-rate 1e-5 --warmup-steps 2000 \
    --weight-decay 0.05 --eval-every 5000 --checkpoint-every 5000 \
    --resume >>"$ROOT/${name}.log" 2>&1
  assert_source_tree_frozen
}

# Paired source/RNG control, exponent-only replacement, U4-magnitude
# compression, strict no-norm counterfactual, then the independent final-norm
# ablation.  Every row must reach 50k before an effectiveness claim.
run_variant fdlogic_wmag7_rms_seed42 7 rms_lut same
run_variant fdlogic_wmag7_shift_rms_seed42 7 shift_rms same
run_variant fdlogic_wmag4_rms_seed42 4 rms_lut same
run_variant fdlogic_wmag7_requant_seed42 7 requant same
run_variant fdlogic_wmag7_none_seed42 7 none same
run_variant fdlogic_wmag7_shift_finalnone_seed42 7 shift_rms none
run_variant fdlogic_wmag7_rms_finalnone_seed42 7 rms_lut none
