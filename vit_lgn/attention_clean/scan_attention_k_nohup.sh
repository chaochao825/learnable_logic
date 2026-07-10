#!/usr/bin/env bash

set -euo pipefail

PYTHON_CMD=${PYTHON_CMD:-python}
SCRIPT=${SCRIPT:-train_logic_vit_tiny.py}
GPU_IDS_STRING=${GPU_IDS:-"0 1"}
read -r -a GPU_IDS <<< "${GPU_IDS_STRING}"

running_pids=()
running_slots=()

pick_free_slot() {
  local slot used
  for slot in "${!GPU_IDS[@]}"; do
    for used in "${running_slots[@]:-}"; do
      if [[ "${used}" == "${slot}" ]]; then
        continue 2
      fi
    done
    printf '%s\n' "${slot}"
    return 0
  done
  return 1
}

prune_finished_jobs() {
  local new_pids=()
  local new_slots=()
  local index

  for index in "${!running_pids[@]}"; do
    if kill -0 "${running_pids[$index]}" 2>/dev/null; then
      new_pids+=("${running_pids[$index]}")
      new_slots+=("${running_slots[$index]}")
    fi
  done

  running_pids=("${new_pids[@]}")
  running_slots=("${new_slots[@]}")
}

wait_for_slot() {
  wait -n
  prune_finished_jobs
}

launch_job() {
  local k="$1"
  local slot="$2"
  local gpu_id="${GPU_IDS[$slot]}"
  local log_file="train_k${k}.log"

  echo "[launch] slot=${slot} gpu=${gpu_id} nohup ${PYTHON_CMD} ${SCRIPT} --attention-k ${k} --majority-train-temperature 1.0 --majority-train-temp-max 1.0 --majority-train-temp-min 1.0 --topk-impl torch-topk --fast-vote > ${log_file} 2>&1 &"
  CUDA_VISIBLE_DEVICES="${gpu_id}" nohup "${PYTHON_CMD}" "${SCRIPT}" --attention-k "${k}" --majority-train-temperature 1.0 --majority-train-temp-max 1.0 --majority-train-temp-min 1.0 --topk-impl torch-topk --fast-vote > "${log_file}" 2>&1 &
  running_pids+=("$!")
  running_slots+=("${slot}")
}

for k in $(seq 41 2 63); do
  while ! slot=$(pick_free_slot); do
    wait_for_slot
  done
  launch_job "${k}" "${slot}"
done

while ((${#running_pids[@]} > 0)); do
  wait_for_slot
done

echo "All jobs launched. Check logs like train_k3.log ... train_k25.log"
