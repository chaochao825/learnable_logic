"""Structural diagnostics for strict character-model LUT payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from vit_lgn.bitplane_lut.analyze_cifar100_payloads import (
    analyze_run,
    write_summary,
)


RUN_PATTERN = re.compile(
    r"^(?:smoke|full|support|repeat)-(?:mixed|causal)-"
    r"v(?:8|32|64|128)-d(?:1|2|4)-s\d+$"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        analyze_run(run_dir)
        for run_dir in sorted(args.runs_root.iterdir())
        if run_dir.is_dir()
        and RUN_PATTERN.fullmatch(run_dir.name)
        and (run_dir / "result.json").is_file()
    ]
    if not rows:
        raise RuntimeError("no completed character-model payloads found")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_summary(args.out, rows)
    print(json.dumps({"runs": len(rows), "output": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
