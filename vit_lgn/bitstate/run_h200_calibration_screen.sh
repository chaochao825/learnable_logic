#!/usr/bin/env bash
set -euo pipefail

group="${1:?usage: run_h200_calibration_screen.sh gpu2|gpu3 [output-root]}"
output_root="${2:-remote_runs/h200_calibration_screen_20260723}"
python_bin="${PYTHON_BIN:-/home/wangmeiqi/miniconda3/envs/lgn/bin/python}"
data_root="${CIFAR10_ROOT:-/home/wangmeiqi/phz/attention-clean/data/cifar-10}"
teacher_root="${TEACHER_ROOT:-/home/wangmeiqi/phz/attention-clean}"
teacher_checkpoint="${TEACHER_CHECKPOINT:-${teacher_root}/logs/20260413_215439/checkpoints/checkpoint_best.pt}"

common=(
  --dataset cifar10
  --data-root "${data_root}"
  --method progressive_hard_st
  --epochs 8
  --validation-size 2000
  --train-limit 10000
  --eval-limit 2000
  --batch-size 128
  --workers 4
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
  --learning-rate 0.002
  --min-learning-rate 0.0001
  --warmup-epochs 1
  --weight-decay 0.0001
  --label-smoothing 0.1
  --tau-start 3.0
  --tau-end 0.5
  --soft-warmup-epochs 2
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

teacher=(
  --teacher-source-dir "${teacher_root}"
  --teacher-checkpoint "${teacher_checkpoint}"
  --teacher-temperature 2.0
)

case "${group}" in
  gpu2)
    run_config tau4 --group-sum-temperature 4
    run_config tau8 --group-sum-temperature 8
    run_config tau8_teacher025 --group-sum-temperature 8 \
      "${teacher[@]}" --teacher-alpha 0.25
    ;;
  gpu3)
    run_config tau16 --group-sum-temperature 16
    run_config tau8_teacher050 --group-sum-temperature 8 \
      "${teacher[@]}" --teacher-alpha 0.5
    run_config tau8_soft4 --group-sum-temperature 8 --soft-warmup-epochs 4
    ;;
  *)
    echo "unknown group: ${group}" >&2
    exit 2
    ;;
esac
