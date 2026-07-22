#!/usr/bin/env bash
set -euo pipefail

group="${1:?usage: run_h200_commitment_screen.sh gpu2|gpu3|ramp010|ramp050 [output-root]}"
output_root="${2:-remote_runs/h200_commitment_screen_20260723}"
python_bin="${PYTHON_BIN:-/home/wangmeiqi/miniconda3/envs/lgn/bin/python}"
data_root="${CIFAR10_ROOT:-/home/wangmeiqi/phz/attention-clean/data/cifar-10}"

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
  --gate-init-strength 0.1
  --local-depth 2
  --global-depth 2
  --heads 8
  --qk-bits 16
  --topk 8
  --votes-per-class 32
  --group-sum-temperature 4
  --learning-rate 0.002
  --min-learning-rate 0.0001
  --warmup-epochs 1
  --weight-decay 0.0001
  --label-smoothing 0.1
  --tau-start 3.0
  --tau-end 0.5
  --soft-warmup-epochs 4
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
  gpu2)
    run_config scale2 --hardening-logit-scale 2
    run_config scale4 --hardening-logit-scale 4
    run_config scale8 --hardening-logit-scale 8
    run_config scale16 --hardening-logit-scale 16
    ;;
  gpu3)
    run_config entropy005 --gate-entropy-weight 0.05
    run_config entropy010 --gate-entropy-weight 0.1
    run_config scale4_ramp010 --hardening-logit-scale 4 \
      --gate-entropy-weight-start 0 --gate-entropy-weight-end 0.1
    run_config scale4_ramp050 --hardening-logit-scale 4 \
      --gate-entropy-weight-start 0 --gate-entropy-weight-end 0.5
    ;;
  ramp010)
    run_config scale4_ramp010 --hardening-logit-scale 4 \
      --gate-entropy-weight-start 0 --gate-entropy-weight-end 0.1
    ;;
  ramp050)
    run_config scale4_ramp050 --hardening-logit-scale 4 \
      --gate-entropy-weight-start 0 --gate-entropy-weight-end 0.5
    ;;
  *)
    echo "unknown group: ${group}" >&2
    exit 2
    ;;
esac
