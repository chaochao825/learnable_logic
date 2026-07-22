#!/usr/bin/env bash
set -euo pipefail

group="${1:?usage: run_gumbel_tuning_screen.sh gpu0|gpu1|paper0|paper1 [output-root]}"
output_root="${2:-remote_runs/gumbel_tuning_screen_20260723}"
python_bin="${PYTHON_BIN:-/home/wangmeiqi/miniconda3/envs/lgn/bin/python}"
data_root="${CIFAR10_ROOT:-/home/wangmeiqi/phz/attention-clean/data/cifar-10}"

common=(
  --dataset cifar10
  --data-root "${data_root}"
  --method gumbel_st
  --epochs 8
  --validation-size 2000
  --train-limit 10000
  --eval-limit 2000
  --batch-size 128
  --workers 2
  --patch-size 4
  --threshold-levels 4
  --state-width 512
  --encoder-kind redundant_predicate
  --predicate-fanin 9
  --encoder-identity-width 192
  --local-depth 2
  --global-depth 2
  --heads 8
  --qk-bits 16
  --topk 8
  --votes-per-class 32
  --group-sum-temperature 4
  --optimizer adamw
  --learning-rate 0.002
  --min-learning-rate 0.0001
  --warmup-epochs 1
  --weight-decay 0.0001
  --label-smoothing 0.1
  --tau 1.0
  --state-balance-weight 0.05
  --state-diversity-weight 0.02
  --state-flip-weight 0.02
  --gate-entropy-weight 0.01
  --target-accuracy 0.5
  --seed 0
  --augment
  --amp-bfloat16
)

run_config() {
  local name="$1"
  shift
  "${python_bin}" -m vit_lgn.bitstate.train_bitstate \
    "${common[@]}" \
    --output-dir "${output_root}/${name}" \
    "$@"
}

case "${group}" in
  gpu0)
    run_config init025_tau100 --gate-init-strength 0.25
    run_config init050_tau100 --gate-init-strength 0.5
    run_config init100_tau100 --gate-init-strength 1.0
    run_config init200_tau100 --gate-init-strength 2.0
    ;;
  gpu1)
    run_config init050_tau050 --gate-init-strength 0.5 --tau 0.5
    run_config init100_tau050 --gate-init-strength 1.0 --tau 0.5
    run_config init200_tau050 --gate-init-strength 2.0 --tau 0.5
    run_config paper_adam_init100_tau100 \
      --gate-init-strength 1.0 \
      --optimizer adam \
      --learning-rate 0.01 \
      --min-learning-rate 0.0001 \
      --weight-decay 0
    ;;
  paper0)
    run_config normal_adam_tau010 \
      --gate-init-mode normal --gate-init-normal-std 1 \
      --optimizer adam --learning-rate 0.01 --lr-schedule constant --weight-decay 0 \
      --tau 0.1 --label-smoothing 0 \
      --state-balance-weight 0 --state-diversity-weight 0 \
      --state-flip-weight 0 --gate-entropy-weight 0
    run_config normal_adam_tau025 \
      --gate-init-mode normal --gate-init-normal-std 1 \
      --optimizer adam --learning-rate 0.01 --lr-schedule constant --weight-decay 0 \
      --tau 0.25 --label-smoothing 0 \
      --state-balance-weight 0 --state-diversity-weight 0 \
      --state-flip-weight 0 --gate-entropy-weight 0
    ;;
  paper1)
    run_config normal_adam_tau050 \
      --gate-init-mode normal --gate-init-normal-std 1 \
      --optimizer adam --learning-rate 0.01 --lr-schedule constant --weight-decay 0 \
      --tau 0.5 --label-smoothing 0 \
      --state-balance-weight 0 --state-diversity-weight 0 \
      --state-flip-weight 0 --gate-entropy-weight 0
    run_config normal_adam_tau100 \
      --gate-init-mode normal --gate-init-normal-std 1 \
      --optimizer adam --learning-rate 0.01 --lr-schedule constant --weight-decay 0 \
      --tau 1.0 --label-smoothing 0 \
      --state-balance-weight 0 --state-diversity-weight 0 \
      --state-flip-weight 0 --gate-entropy-weight 0
    ;;
  *)
    echo "unknown group: ${group}" >&2
    exit 2
    ;;
esac
