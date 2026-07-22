#!/usr/bin/env bash
set -euo pipefail

group="${1:?usage: run_h200_screen.sh gpu2|gpu3 [output-root]}"
output_root="${2:-remote_runs/h200_screen_20260723}"
python_bin="${PYTHON_BIN:-/home/wangmeiqi/miniconda3/envs/lgn/bin/python}"
data_root="${CIFAR10_ROOT:-/home/wangmeiqi/phz/attention-clean/data/cifar-10}"

common=(
  --dataset cifar10
  --data-root "${data_root}"
  --epochs 8
  --validation-size 2000
  --train-limit 10000
  --eval-limit 2000
  --batch-size 128
  --workers 4
  --patch-size 4
  --threshold-levels 4
  --state-width 512
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
  --target-accuracy 0.5
  --seed 0
  --augment
  --amp-bfloat16
)

run_config() {
  local name="$1"
  local method="$2"
  local encoder="$3"
  shift 3
  "${python_bin}" -m vit_lgn.bitstate.train_bitstate \
    "${common[@]}" \
    --method "${method}" \
    --encoder-kind "${encoder}" \
    --output-dir "${output_root}/${name}" \
    "$@"
}

predicate_encoder=(
  --predicate-fanin 9
  --encoder-identity-width 192
  --predicate-temperature 1.0
)

anti_collapse=(
  --state-balance-weight 0.05
  --state-diversity-weight 0.02
  --state-flip-weight 0.02
  --gate-entropy-weight 0.01
  --gate-entropy-target-start 0.8
  --gate-entropy-target-end 0.1
)

case "${group}" in
  gpu2)
    run_config direct_gumbel gumbel_st thermometer
    run_config predicate_gumbel gumbel_st redundant_predicate "${predicate_encoder[@]}"
    run_config predicate_progressive_reg progressive_hard_st redundant_predicate \
      "${predicate_encoder[@]}" "${anti_collapse[@]}"
    ;;
  gpu3)
    run_config direct_progressive progressive_hard_st thermometer
    run_config predicate_progressive progressive_hard_st redundant_predicate \
      "${predicate_encoder[@]}"
    run_config predicate_hard_reg hard_st redundant_predicate \
      "${predicate_encoder[@]}" "${anti_collapse[@]}"
    ;;
  *)
    echo "unknown group: ${group}" >&2
    exit 2
    ;;
esac
