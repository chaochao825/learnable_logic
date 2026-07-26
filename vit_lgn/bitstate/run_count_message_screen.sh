#!/usr/bin/env bash
set -euo pipefail

variant="${1:?usage: run_count_message_screen.sh majority|count seed [output-root]}"
seed="${2:?usage: run_count_message_screen.sh majority|count seed [output-root]}"
output_root="${3:-remote_runs/count_message_screen_20260723}"
python_bin="${PYTHON_BIN:-/home/wangmeiqi/miniconda3/envs/lgn/bin/python}"
data_root="${CIFAR10_ROOT:-/home/wangmeiqi/phz/attention-clean/data/cifar-10}"

case "${variant}" in
  majority)
    method_id="bitstate_progressive"
    message_args=(--message-mode majority)
    ;;
  count)
    method_id="bitstate_count_message_v0"
    message_args=(
      --message-mode count_threshold_hybrid
      --message-count-thresholds 1,3,5,7
      --message-count-fraction 0.25
    )
    ;;
  *)
    echo "unknown variant: ${variant}" >&2
    exit 2
    ;;
esac

run_dir="${output_root}/${variant}_seed${seed}"
exec "${python_bin}" -m vit_lgn.bitstate.train_bitstate \
  --dataset cifar10 \
  --data-root "${data_root}" \
  --epochs 8 \
  --validation-size 2000 \
  --train-limit 10000 \
  --eval-limit 2000 \
  --batch-size 128 \
  --workers 8 \
  --patch-size 4 \
  --threshold-levels 4 \
  --state-width 4096 \
  --encoder-kind redundant_predicate \
  --predicate-fanin 9 \
  --predicate-chunk-size 1024 \
  --encoder-identity-width 192 \
  --global-token-mode majority \
  --gate-init-strength 0.1 \
  --local-depth 2 \
  --global-depth 2 \
  --heads 8 \
  --qk-bits 64 \
  --topk 8 \
  --votes-per-class 64 \
  --group-sum-temperature 8 \
  --method progressive_hard_st \
  --hardening-logit-scale 16 \
  --learning-rate 0.002 \
  --min-learning-rate 0.00005 \
  --warmup-epochs 2 \
  --lr-schedule cosine \
  --optimizer adamw \
  --weight-decay 0.0001 \
  --grad-clip 1 \
  --label-smoothing 0.1 \
  --tau 1 \
  --tau-start 3 \
  --tau-end 0.5 \
  --soft-warmup-epochs 4 \
  --state-balance-weight 0.05 \
  --state-diversity-weight 0.02 \
  --state-flip-weight 0.02 \
  --gate-entropy-weight 0.01 \
  --gate-entropy-target-start 0.8 \
  --gate-entropy-target-end 0.1 \
  --target-accuracy 0.2 \
  --inactive-batches 8 \
  --seed "${seed}" \
  --augment \
  --amp-bfloat16 \
  --save-checkpoint \
  --registry-method-id "${method_id}" \
  --protocol-id bitstate_count_message_10k_8e_s012 \
  "${message_args[@]}" \
  --output-dir "${run_dir}"
