#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 10 ]]; then
  echo "usage: $0 RUN_ID GPU POLICY VOTES_PER_CLASS BLOCKS EPOCHS MIN_EPOCHS PATIENCE MAX_TRAIN SEED" >&2
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
max_train=$9
seed=${10}

project_root=${BITPLANE_PROJECT_ROOT:-/home/wangmeiqi/bitplane_lut_hardlgn_20260724_v1}
run_root=${BITPLANE_CIFAR100_RUN_ROOT:-/home/wangmeiqi/bitplane_lut_cifar100_20260724_v1}
python_bin=${BITPLANE_PYTHON:-/home/wangmeiqi/miniconda3/envs/lgn/bin/python}
data_root=${BITPLANE_CIFAR100_DATA_ROOT:-${project_root}/data}
protocol=${project_root}/docs/protocols/bitplane_lut_cifar100_scale_v1_20260724.md
out_dir=${run_root}/runs/${run_id}
log_dir=${run_root}/logs
status_dir=${run_root}/status

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

"${python_bin}" -m vit_lgn.bitplane_lut.train_cifar100 \
  --out-dir "${out_dir}" \
  --data-root "${data_root}" \
  --protocol-path "${protocol}" \
  --seed "${seed}" \
  --votes-per-class "${votes_per_class}" \
  --blocks "${blocks}" \
  --candidate-policy "${policy}" \
  --refit-mode argmax \
  --max-train-samples "${max_train}" \
  --calibration-samples 5000 \
  --epochs-per-block "${epochs}" \
  --minimum-epochs "${minimum_epochs}" \
  --patience "${patience}" \
  --batch-size 128 \
  --eval-batch-size 128 \
  --diagnostic-rows 512 \
  --device cuda \
  2>&1 | tee "${log_dir}/${run_id}.log"
