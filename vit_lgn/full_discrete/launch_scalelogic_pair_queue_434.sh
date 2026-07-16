#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data2/wangmeiqi/learnable_logic_hadamard_mixer_20260716}
GPU_INDEX=${GPU_INDEX:-2}
CANDIDATE_EXIT=$ROOT/scalelogic_d12e384_h12_local4_seed42_50k.exit
CONTROL_LOG=$ROOT/scalelogic_d12e384_h12_local0_seed42_50k.log
CONTROL_EXIT=$ROOT/scalelogic_d12e384_h12_local0_seed42_50k.exit

while [[ ! -f "$CANDIDATE_EXIT" ]]; do
  sleep 60
done
if [[ $(<"$CANDIDATE_EXIT") != 0 ]]; then
  echo "Candidate failed; paired control will not start" >&2
  exit 4
fi

while true; do
  read -r used util < <(
    nvidia-smi --id="$GPU_INDEX" \
      --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits \
      | tr -d ' ' | tr ',' ' '
  )
  (( used <= 1024 && util <= 10 )) && break
  sleep 60
done

set +e
LOCAL_LAYERS=0 \
RUN_NAME=scalelogic_d12e384_h12_local0_seed42_50k \
GPU_INDEX="$GPU_INDEX" \
bash "$ROOT/vit_lgn/full_discrete/launch_scalelogic_50k_434.sh" \
  >"$CONTROL_LOG" 2>&1
code=$?
set -e
printf '%s\n' "$code" >"$CONTROL_EXIT"
exit "$code"
