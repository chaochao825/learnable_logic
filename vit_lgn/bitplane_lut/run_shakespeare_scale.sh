#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 10 ]]; then
  echo "usage: $0 RUN_ID GPU POLICY VOTES_PER_CLASS BLOCKS EPOCHS MIN_EPOCHS PATIENCE TRAIN_SAMPLES SEED" >&2
  exit 2
fi

run_id=$1
gpu=$2
policy=$3
votes_per_class=$4
blocks=$5
epochs=$6
minimum_epochs=$7
patience=$8
train_samples=$9
seed=${10}

project_root=${BITPLANE_PROJECT_ROOT:-/home/wangmeiqi/bitplane_lut_sequence_20260724_v1}
run_root=${BITPLANE_SEQUENCE_RUN_ROOT:-/home/wangmeiqi/bitplane_lut_shakespeare_20260724_v1}
python_bin=${BITPLANE_PYTHON:-/home/wangmeiqi/bitplane_lut_cifar100_env_20260724/bin/python}
corpus=${BITPLANE_SHAKESPEARE_CORPUS:-${project_root}/data/tinyshakespeare.txt}
protocol=${project_root}/docs/protocols/bitplane_lut_shakespeare_char_scale_v1_20260724.md
out_dir=${run_root}/runs/${run_id}
log_dir=${run_root}/logs
status_dir=${run_root}/status
prefix_run_dir=${BITPLANE_PREFIX_RUN_DIR:-}

prefix_args=()
if [[ -n "${prefix_run_dir}" ]]; then
  prefix_args=(--prefix-run-dir "${prefix_run_dir}")
fi

mkdir -p "${log_dir}" "${status_dir}"
if [[ -e "${out_dir}" ]]; then
  echo "refusing to reuse output directory: ${out_dir}" >&2
  exit 3
fi

on_exit() {
  code=$?
  printf '%s\n' "${code}" > "${status_dir}/${run_id}.exit"
}
trap on_exit EXIT

cd "${project_root}"
export CUDA_VISIBLE_DEVICES=${gpu}
export PYTHONUNBUFFERED=1

"${python_bin}" -m vit_lgn.bitplane_lut.train_shakespeare \
  --out-dir "${out_dir}" \
  --corpus-path "${corpus}" \
  --protocol-path "${protocol}" \
  "${prefix_args[@]}" \
  --seed "${seed}" \
  --context-length 64 \
  --votes-per-class "${votes_per_class}" \
  --blocks "${blocks}" \
  --candidate-policy "${policy}" \
  --train-samples "${train_samples}" \
  --validation-samples 20000 \
  --test-samples 0 \
  --calibration-samples 5000 \
  --epochs-per-block "${epochs}" \
  --minimum-epochs "${minimum_epochs}" \
  --patience "${patience}" \
  --batch-size 256 \
  --eval-batch-size 256 \
  --diagnostic-rows 512 \
  --device cuda \
  2>&1 | tee "${log_dir}/${run_id}.log"
