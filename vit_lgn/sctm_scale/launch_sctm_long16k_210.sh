#!/usr/bin/env bash
set -euo pipefail

cd /home/spco/sow_linear/ViT-LGN_goal6plus_sctm_scale_20260628
unset PREFIX || true

PY=/home/wangmeiqi/anaconda3/envs/vit310/bin/python
export PYTHONPATH=.:/home/wangmeiqi/pyf

LOGDIR=runs/sctm_long16k_20260629_logs
mkdir -p "${LOGDIR}"

common=(
  --mixer sctm_lowbit_weighted
  --sctm-topk 8
  --sctm-weight-bits 2
  --sctm-patch-path local3x3
  --patch-size 4
  --logic-mlp-ratio 1
  --augment
  --teacher-iters 16000
)

launch() {
  local gpu="$1"
  local name="$2"
  shift 2
  local log="${LOGDIR}/${name}.log"
  local pidfile="${LOGDIR}/${name}.pid"
  nohup env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${PYTHONPATH}" "${PY}" sctm_token_mixer_experiment.py "$@" >"${log}" 2>&1 &
  local pid="$!"
  echo "${pid}" >"${pidfile}"
  echo "${name} pid=${pid} gpu=${gpu} log=${log}"
}

launch 1 d16_e1024_h32_aug16k \
  "${common[@]}" \
  --depth 16 \
  --embed-dim 1024 \
  --num-heads 32 \
  --out-dir runs/sctm_long16k_d16_e1024_h32_k8_local3x3_aug_20260629

launch 2 d12_e1536_h48_aug16k \
  "${common[@]}" \
  --depth 12 \
  --embed-dim 1536 \
  --num-heads 48 \
  --out-dir runs/sctm_long16k_d12_e1536_h48_k8_local3x3_aug_20260629

launch 3 d20_e1024_h32_aug16k \
  "${common[@]}" \
  --depth 20 \
  --embed-dim 1024 \
  --num-heads 32 \
  --out-dir runs/sctm_long16k_d20_e1024_h32_k8_local3x3_aug_20260629
