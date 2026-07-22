#!/usr/bin/env bash
set -euo pipefail

variant="${1:?usage: run_h200_long_cifar.sh dlgn|anneal|gumbel|dlgn_normal_adam|anneal_normal_adam|gumbel_normal_adam_tau010|gumbel_normal_adam_tau100|hard_scale4|hard_scale16|hard_scale4_ramp050 [output-root]}"
output_root="${2:-remote_runs/h200_long_cifar_20260723}"
python_bin="${PYTHON_BIN:-/home/wangmeiqi/miniconda3/envs/lgn/bin/python}"
data_root="${CIFAR10_ROOT:-/home/wangmeiqi/phz/attention-clean/data/cifar-10}"

epochs="${EPOCHS:-30}"
validation_size="${VALIDATION_SIZE:-5000}"
train_limit="${TRAIN_LIMIT:-0}"
eval_limit="${EVAL_LIMIT:-0}"
batch_size="${BATCH_SIZE:-128}"
workers="${WORKERS:-8}"
state_width="${STATE_WIDTH:-4096}"
identity_width="${IDENTITY_WIDTH:-192}"
qk_bits="${QK_BITS:-64}"
votes_per_class="${VOTES_PER_CLASS:-64}"
group_sum_temperature="${GROUP_SUM_TEMPERATURE:-8}"
seed="${SEED:-0}"

common=(
  --dataset cifar10
  --data-root "${data_root}"
  --epochs "${epochs}"
  --validation-size "${validation_size}"
  --train-limit "${train_limit}"
  --eval-limit "${eval_limit}"
  --batch-size "${batch_size}"
  --workers "${workers}"
  --patch-size 4
  --threshold-levels 4
  --state-width "${state_width}"
  --encoder-kind redundant_predicate
  --predicate-fanin 9
  --predicate-chunk-size 1024
  --encoder-identity-width "${identity_width}"
  --gate-init-strength 0.1
  --local-depth 2
  --global-depth 2
  --heads 8
  --qk-bits "${qk_bits}"
  --topk 8
  --votes-per-class "${votes_per_class}"
  --group-sum-temperature "${group_sum_temperature}"
  --learning-rate 0.002
  --min-learning-rate 0.00005
  --warmup-epochs 2
  --weight-decay 0.0001
  --label-smoothing 0.1
  --tau 1.0
  --tau-start 3.0
  --tau-end 0.5
  --soft-warmup-epochs 15
  --state-balance-weight 0.05
  --state-diversity-weight 0.02
  --state-flip-weight 0.02
  --gate-entropy-weight 0.01
  --target-accuracy 0.2
  --inactive-batches 8
  --seed "${seed}"
  --augment
  --amp-bfloat16
  --save-checkpoint
)

variant_args=()
case "${variant}" in
  dlgn)
    variant_args=(--method soft)
    ;;
  anneal)
    variant_args=(--method anneal)
    ;;
  gumbel)
    variant_args=(--method gumbel_st)
    ;;
  dlgn_normal_adam)
    variant_args=(
      --method soft
      --gate-init-mode normal
      --gate-init-normal-std 1
      --optimizer adam
      --learning-rate 0.01
      --lr-schedule constant
      --weight-decay 0
      --label-smoothing 0
      --state-balance-weight 0
      --state-diversity-weight 0
      --state-flip-weight 0
      --gate-entropy-weight 0
    )
    ;;
  anneal_normal_adam)
    variant_args=(
      --method anneal
      --gate-init-mode normal
      --gate-init-normal-std 1
      --optimizer adam
      --learning-rate 0.01
      --lr-schedule constant
      --weight-decay 0
      --label-smoothing 0
      --state-balance-weight 0
      --state-diversity-weight 0
      --state-flip-weight 0
      --gate-entropy-weight 0
    )
    ;;
  gumbel_normal_adam_tau010)
    variant_args=(
      --method gumbel_st
      --gate-init-mode normal
      --gate-init-normal-std 1
      --optimizer adam
      --learning-rate 0.01
      --lr-schedule constant
      --weight-decay 0
      --label-smoothing 0
      --tau 0.1
      --state-balance-weight 0
      --state-diversity-weight 0
      --state-flip-weight 0
      --gate-entropy-weight 0
    )
    ;;
  gumbel_normal_adam_tau100)
    variant_args=(
      --method gumbel_st
      --gate-init-mode normal
      --gate-init-normal-std 1
      --optimizer adam
      --learning-rate 0.01
      --lr-schedule constant
      --weight-decay 0
      --label-smoothing 0
      --tau 1.0
      --state-balance-weight 0
      --state-diversity-weight 0
      --state-flip-weight 0
      --gate-entropy-weight 0
    )
    ;;
  hard_scale4)
    variant_args=(--method progressive_hard_st --hardening-logit-scale 4)
    ;;
  hard_scale16)
    variant_args=(--method progressive_hard_st --hardening-logit-scale 16)
    ;;
  hard_scale4_ramp050)
    variant_args=(
      --method progressive_hard_st
      --hardening-logit-scale 4
      --gate-entropy-weight-start 0
      --gate-entropy-weight-end 0.5
    )
    ;;
  *)
    echo "unknown variant: ${variant}" >&2
    exit 2
    ;;
esac

run_dir="${output_root}/${variant}_w${state_width}_seed${seed}"
command=(
  "${python_bin}"
  -m vit_lgn.bitstate.train_bitstate
  "${common[@]}"
  "${variant_args[@]}"
  --output-dir "${run_dir}"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

"${command[@]}"
