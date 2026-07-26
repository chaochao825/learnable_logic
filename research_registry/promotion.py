from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PromotionDecision:
    passed: bool
    reasons: tuple[str, ...]
    hard_acc_delta: float | None
    acc_gap_delta: float | None
    seed_wins: int

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "hard_acc_delta": self.hard_acc_delta,
            "acc_gap_delta": self.acc_gap_delta,
            "seed_wins": self.seed_wins,
        }


def _number(row: Mapping[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise ValueError(f"missing {key} in {row.get('result_id', '<unknown>')}")
    return float(value)


def _rows_by_seed(rows: Iterable[Mapping[str, str]]) -> dict[int, Mapping[str, str]]:
    output: dict[int, Mapping[str, str]] = {}
    for row in rows:
        seed_value = row.get("seed", "")
        if seed_value == "":
            continue
        seed = int(seed_value)
        if seed in output:
            raise ValueError(f"duplicate seed {seed}")
        output[seed] = row
    return output


def compare_candidate(
    candidate_rows: Iterable[Mapping[str, str]],
    baseline_rows: Iterable[Mapping[str, str]],
    policy: Mapping[str, object],
) -> PromotionDecision:
    candidate = _rows_by_seed(candidate_rows)
    baseline = _rows_by_seed(baseline_rows)
    reasons: list[str] = []
    shared = sorted(set(candidate) & set(baseline))
    minimum_seeds = int(policy["minimum_seeds"])
    if len(shared) < minimum_seeds:
        reasons.append(f"requires {minimum_seeds} paired seeds; found {len(shared)}")
    if set(candidate) != set(baseline):
        reasons.append("candidate and baseline seed sets differ")
    if not shared:
        return PromotionDecision(False, tuple(reasons), None, None, 0)

    for seed in shared:
        candidate_row = candidate[seed]
        baseline_row = baseline[seed]
        if candidate_row.get("protocol_id") != baseline_row.get("protocol_id"):
            reasons.append(f"seed {seed} uses an unmatched protocol")
        if candidate_row.get("dataset") != baseline_row.get("dataset"):
            reasons.append(f"seed {seed} uses an unmatched dataset")
        candidate_protocol_hash = candidate_row.get("protocol_sha256", "")
        baseline_protocol_hash = baseline_row.get("protocol_sha256", "")
        if len(candidate_protocol_hash) != 64:
            reasons.append(f"seed {seed} lacks a full protocol hash")
        if candidate_protocol_hash != baseline_protocol_hash:
            reasons.append(f"seed {seed} protocol hashes differ")
        for role, row in (("candidate", candidate_row), ("baseline", baseline_row)):
            if row.get("selection_split") != "validation":
                reasons.append(f"seed {seed} {role} was not validation-selected")
            if row.get("training_health") != "pass":
                reasons.append(f"seed {seed} {role} training health is not pass")
            if policy.get("require_finite_curve"):
                if not row.get("best_hard_acc") or not row.get("final_hard_acc"):
                    reasons.append(
                        f"seed {seed} {role} lacks best/final validation metrics"
                    )
                else:
                    late_drop = _number(row, "best_hard_acc") - _number(
                        row, "final_hard_acc"
                    )
                    if late_drop > float(policy["maximum_late_validation_drop"]):
                        reasons.append(
                            f"seed {seed} {role} late validation drop is "
                            f"{late_drop:.6f}"
                        )
            required_runtime = str(policy["required_runtime_compliance"])
            if row.get("runtime_compliance") != required_runtime:
                reasons.append(f"seed {seed} {role} lacks {required_runtime}")
            if row.get("float_tensor_count") != "0":
                reasons.append(
                    f"seed {seed} {role} did not prove zero floating tensors"
                )
            if int(row.get("audit_ops", "") or 0) <= 0:
                reasons.append(f"seed {seed} {role} lacks an operator audit count")
            if policy.get("require_deployment_payload_hash") and len(
                row.get("deployment_payload_sha256", "")
            ) != 64:
                reasons.append(
                    f"seed {seed} {role} lacks a standalone deployment hash"
                )

    hard_deltas = [
        _number(candidate[seed], "hard_acc") - _number(baseline[seed], "hard_acc")
        for seed in shared
    ]
    gap_deltas = [
        _number(candidate[seed], "acc_gap") - _number(baseline[seed], "acc_gap")
        for seed in shared
    ]
    mean_hard_delta = mean(hard_deltas)
    mean_gap_delta = mean(gap_deltas)
    seed_wins = sum(delta > 0.0 for delta in hard_deltas)
    minimum_delta = float(policy["minimum_mean_hard_acc_delta"])
    if mean_hard_delta < minimum_delta:
        reasons.append(
            f"mean hard-accuracy delta is {mean_hard_delta:.6f}; "
            f"requires {minimum_delta:.6f}"
        )
    if min(hard_deltas) < -float(policy["maximum_single_seed_hard_acc_regression"]):
        reasons.append("at least one seed exceeds the hard-accuracy regression limit")
    if seed_wins < int(policy["minimum_seed_wins"]):
        reasons.append(f"only {seed_wins} paired seeds improve")
    if mean_gap_delta > float(policy["maximum_mean_acc_gap_increase"]):
        reasons.append(f"mean accuracy-gap increase is {mean_gap_delta:.6f}")
    return PromotionDecision(
        passed=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
        hard_acc_delta=mean_hard_delta,
        acc_gap_delta=mean_gap_delta,
        seed_wins=seed_wins,
    )


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_results(
    rows: Iterable[Mapping[str, str]],
    *,
    method_id: str,
    protocol_id: str,
    variant: str = "",
) -> list[Mapping[str, str]]:
    return [
        row
        for row in rows
        if row.get("method_id") == method_id
        and row.get("protocol_id") == protocol_id
        and (not variant or variant in row.get("result_id", ""))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a research-method promotion")
    parser.add_argument("--candidate", required=True, help="candidate method_id")
    parser.add_argument("--baseline", required=True, help="baseline method_id")
    parser.add_argument("--protocol", required=True)
    parser.add_argument(
        "--candidate-variant",
        default="",
        help="optional token that must occur in candidate result_id",
    )
    parser.add_argument(
        "--baseline-variant",
        default="",
        help="optional token that must occur in baseline result_id",
    )
    args = parser.parse_args()
    rows = _load_rows(ROOT / "research_registry" / "results.csv")
    policy = json.loads(
        (ROOT / "research_registry" / "promotion_policy.json").read_text()
    )
    decision = compare_candidate(
        select_results(
            rows,
            method_id=args.candidate,
            protocol_id=args.protocol,
            variant=args.candidate_variant,
        ),
        select_results(
            rows,
            method_id=args.baseline,
            protocol_id=args.protocol,
            variant=args.baseline_variant,
        ),
        policy,
    )
    print(json.dumps(decision.to_dict(), indent=2, sort_keys=True))
    raise SystemExit(0 if decision.passed else 1)


if __name__ == "__main__":
    main()
