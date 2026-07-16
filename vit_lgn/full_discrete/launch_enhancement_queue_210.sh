#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/spco/sow_linear/learnable_logic_full_discrete_enhancements_20260713
DATA=/home/spco/sow_linear/ViT-LGN_attention_clean_20260629/data/cifar-10
GPU_INDEX=${GPU_INDEX:-0}
VARIANT_SET=${VARIANT_SET:-all}
case "$VARIANT_SET" in
  core|expert|all) ;;
  *)
    echo "Unknown VARIANT_SET=$VARIANT_SET (expected core, expert, or all)" >&2
    exit 2
    ;;
esac
if [[ ! "$GPU_INDEX" =~ ^(0|[1-9][0-9]*)$ ]] || \
   ! nvidia-smi --id="$GPU_INDEX" --query-gpu=index --format=csv,noheader,nounits >/dev/null 2>&1; then
  echo "Invalid GPU_INDEX=$GPU_INDEX" >&2
  exit 2
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

run_variant() (
  local name=$1
  shift
  local output_lock=/tmp/codex_lgn_vit_${name}.lock
  exec 8>"$output_lock"
  if ! flock -n 8; then
    echo "$name is already running" >&2
    exit 3
  fi
  python -m vit_lgn.full_discrete.train_cifar \
    --data-root "$DATA" --out-dir "$ROOT/runs/$name" \
    --seed 42 --steps 50000 --batch-size 128 --eval-batch-size 256 \
    --dim 192 --depth 6 --heads 6 --topk 8 --mlp-ratio 4 \
    --weight-magnitude-bits 7 --activation-bits 8 --qk-lanes 7 \
    --learning-rate 3e-4 --min-learning-rate 1e-5 --warmup-steps 2000 \
    --weight-decay 0.05 --eval-every 5000 --checkpoint-every 5000 \
    --resume "$@" >>"$ROOT/${name}.log" 2>&1
)

run_core_variants() {
  run_variant fd_control_seed42
  run_variant fd_learned_gap_seed42 --learned-gap
  run_variant fd_group_lut_seed42 --group-lut-groups 48
  run_variant fd_local3_seed42 --local-layers 3
}

run_expert_variants() {
  run_variant fd_logicexpert_seed42 --logic-expert-width 1024 --logic-expert-count 2
  run_variant fd_state_dynamic_seed42 --state-control dynamic --state-expert-width 512
  run_variant fd_state_static_seed42 --state-control static --state-expert-width 512
  run_variant fd_state_script_seed42 --state-control script --state-expert-width 512
}

case "$VARIANT_SET" in
  core) run_core_variants ;;
  expert) run_expert_variants ;;
  all)
    run_core_variants
    run_expert_variants
    ;;
esac
