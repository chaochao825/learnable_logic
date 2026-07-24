#!/usr/bin/env bash
set -euo pipefail

project_root=${BITPLANE_PROJECT_ROOT:-/home/wangmeiqi/bitplane_lut_sequence_20260724_v1}
run_root=${BITPLANE_SEQUENCE_RUN_ROOT:-/home/wangmeiqi/bitplane_lut_shakespeare_20260724_v1}
python_bin=${BITPLANE_PYTHON:-/home/wangmeiqi/miniconda3/envs/lgn/bin/python}
gpu=${BITPLANE_GATE_GPU:-0}
poll_seconds=${BITPLANE_GATE_POLL_SECONDS:-60}

baseline_id=full-causal-v64-d2-s0
depth_id=full-causal-v64-d4-s0
support_id=support-causal-v128-d2-s0
target_id=full-causal-v128-d4-s0
status_dir=${run_root}/status
decision_path=${run_root}/logs/${target_id}.gate.json

wait_for_result() {
  local run_id=$1
  local result=${run_root}/runs/${run_id}/result.json
  local status=${status_dir}/${run_id}.exit
  while [[ ! -f "${result}" ]]; do
    if [[ -f "${status}" ]] && [[ $(<"${status}") != 0 ]]; then
      echo "prerequisite failed: ${run_id}" >&2
      exit 5
    fi
    sleep "${poll_seconds}"
  done
}

mkdir -p "${run_root}/logs" "${status_dir}"
wait_for_result "${depth_id}"
wait_for_result "${support_id}"

baseline_result=${run_root}/runs/${baseline_id}/result.json
depth_result=${run_root}/runs/${depth_id}/result.json
support_result=${run_root}/runs/${support_id}/result.json

if "${python_bin}" - \
  "${baseline_result}" \
  "${depth_result}" \
  "${support_result}" \
  "${decision_path}" <<'PY'
import json
from pathlib import Path
import sys

baseline = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
depth = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
support = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
decision_path = Path(sys.argv[4])
trigram = float(
    depth["integer_ngram_references"]["trigram"]["validation"]["hard_acc"]
)


def strict(result: dict[str, object]) -> bool:
    runtime = result["strict_runtime"]
    structure = result["structure"]
    return bool(
        runtime["exact_carrier_logit_match"]
        and int(runtime["float_tensor_count"]) == 0
        and int(structure["learned_dense_integer_matrix_count"]) == 0
        and int(structure["learned_numeric_weight_count"]) == 0
        and result["training_health"]["finite_gradients"]
    )


checks = {
    "depth_beats_v64_d2": float(depth["hard_acc"]) > float(baseline["hard_acc"]),
    "depth_beats_trigram": float(depth["hard_acc"]) > trigram,
    "v128_d2_beats_v64_d2": float(support["hard_acc"])
    > float(baseline["hard_acc"]),
    "depth_reuses_exact_v64_prefix": depth.get("prefix", {}).get(
        "payload_sha256"
    )
    == baseline["hard_payload_sha256"],
    "baseline_strict": strict(baseline),
    "depth_strict": strict(depth),
    "support_strict": strict(support),
}
decision = {
    "allow_v128_d4": all(checks.values()),
    "checks": checks,
    "validation_hard_acc": {
        "v64_d2": baseline["hard_acc"],
        "v64_d4": depth["hard_acc"],
        "v128_d2": support["hard_acc"],
        "trigram": trigram,
    },
    "prefix_payload_sha256": support["hard_payload_sha256"],
}
temporary = decision_path.with_suffix(decision_path.suffix + ".tmp")
temporary.write_text(
    json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
temporary.replace(decision_path)
print(json.dumps(decision, sort_keys=True))
raise SystemExit(0 if decision["allow_v128_d4"] else 10)
PY
then
  :
else
  code=$?
  if [[ ${code} -eq 10 ]]; then
    echo "v128-d4 gate rejected; see ${decision_path}"
    exit 0
  fi
  exit "${code}"
fi

target_dir=${run_root}/runs/${target_id}
if [[ -e "${target_dir}" ]]; then
  echo "refusing to reuse output directory: ${target_dir}" >&2
  exit 3
fi

support_dir=${run_root}/runs/${support_id}
lock_path=${BITPLANE_GATE_LOCK:-/tmp/bitplane-lut-shakespeare-gpu${gpu}.lock}
exec flock -w 600 "${lock_path}" env \
  BITPLANE_PROJECT_ROOT="${project_root}" \
  BITPLANE_SEQUENCE_RUN_ROOT="${run_root}" \
  BITPLANE_PYTHON="${python_bin}" \
  BITPLANE_PREFIX_RUN_DIR="${support_dir}" \
  bash "${project_root}/vit_lgn/bitplane_lut/run_shakespeare_scale.sh" \
    "${target_id}" "${gpu}" sequence_causal 128 4 20 6 4 100000 0
