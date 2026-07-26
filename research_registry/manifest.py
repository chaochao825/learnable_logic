from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping


def canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def source_file_hashes(
    paths: list[Path],
    *,
    base: Path,
) -> dict[str, str]:
    output: dict[str, str] = {}
    for path in paths:
        resolved = path.resolve()
        output[resolved.relative_to(base.resolve()).as_posix()] = hashlib.sha256(
            resolved.read_bytes()
        ).hexdigest()
    return output


def runtime_environment() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch

        payload.update(
            {
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
            }
        )
        if torch.cuda.is_available():
            payload["device"] = torch.cuda.get_device_name(torch.cuda.current_device())
    except ImportError:
        payload["torch"] = None
    return payload


def build_run_manifest(
    *,
    method_id: str,
    protocol_id: str,
    protocol: Mapping[str, Any],
    seed: int,
    source: Mapping[str, Any],
    capacity: Mapping[str, Any],
    training: Mapping[str, Any],
    result: Mapping[str, Any],
    runtime_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "method_id": method_id,
        "protocol_id": protocol_id,
        "protocol_sha256": canonical_sha256(protocol),
        "seed": int(seed),
        "source": dict(source),
        "capacity": dict(capacity),
        "training": dict(training),
        "result": dict(result),
        "runtime_audit": dict(runtime_audit or {}),
        "environment": runtime_environment(),
    }


def write_run_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
