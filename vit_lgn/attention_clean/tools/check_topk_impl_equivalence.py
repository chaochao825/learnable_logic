from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.packed_xnor_topk import PackedXnorTopK


def build_mask_from_indices(indices: torch.Tensor, row_count: int) -> torch.Tensor:
    row_ids = torch.arange(row_count, device=indices.device, dtype=indices.dtype)
    view_shape = [1] * indices.ndim
    view_shape[-1] = row_count
    row_ids = row_ids.view(*view_shape)
    return torch.any(indices.unsqueeze(-1) == row_ids, dim=-2)


def tie_sensitive_rows(scores: torch.Tensor, k: int) -> torch.Tensor:
    width = scores.shape[-1]
    if k <= 0 or k >= width:
        return torch.zeros(scores.shape[:-1], dtype=torch.bool, device=scores.device)

    sorted_scores, _ = torch.sort(scores, dim=-1, descending=True)
    kth = sorted_scores[..., k - 1]
    next_after_k = sorted_scores[..., k]
    return kth == next_after_k


def run_check(batch: int, rows: int, bits: int, topk: int, seeds: int) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, batch={batch}, rows={rows}, bits={bits}, topk={topk}, seeds={seeds}")

    impl_w = PackedXnorTopK(topk_impl="winner-tree").to(device).eval()
    impl_t = PackedXnorTopK(topk_impl="torch-topk").to(device).eval()

    total_rows = 0
    non_tie_rows = 0
    tie_rows = 0
    non_tie_mismatch_rows = 0
    xnor_popcount_mismatch = 0

    selector_matches_selected_is_1_w = 0
    selector_matches_selected_is_0_w = 0
    selector_matches_selected_is_1_t = 0
    selector_matches_selected_is_0_t = 0

    for seed in range(seeds):
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)

        q = torch.randint(0, 2, (batch, rows, bits), generator=g, dtype=torch.int64).to(device=device, dtype=torch.float32)
        k = torch.randint(0, 2, (batch, rows, bits), generator=g, dtype=torch.int64).to(device=device, dtype=torch.float32)

        out_w = impl_w(q, k, topk=topk, return_selector_mask=True, return_topk_packed_scores=False)
        out_t = impl_t(q, k, topk=topk, return_selector_mask=True, return_topk_packed_scores=False)

        if not torch.equal(out_w["xnor_popcount"], out_t["xnor_popcount"]):
            xnor_popcount_mismatch += 1

        scores = out_w["xnor_popcount"]
        k_eff = min(topk, rows)

        mask_w = build_mask_from_indices(out_w["topk_indices"], rows)
        mask_t = build_mask_from_indices(out_t["topk_indices"], rows)

        tie_mask = tie_sensitive_rows(scores, k_eff)
        same_set = torch.equal(mask_w[~tie_mask], mask_t[~tie_mask])
        if not same_set:
            diff_rows = torch.any(mask_w != mask_t, dim=-1) & (~tie_mask)
            non_tie_mismatch_rows += int(diff_rows.sum().item())

        total_rows += tie_mask.numel()
        tie_rows += int(tie_mask.sum().item())
        non_tie_rows += int((~tie_mask).sum().item())

        selector_w = out_w["selector_mask"]
        selector_t = out_t["selector_mask"]

        if selector_w is None or selector_t is None:
            print("selector_mask is None, cannot validate mask semantics")
            return 2

        selector_w_bool = selector_w.to(torch.bool)
        selector_t_bool = selector_t.to(torch.bool)

        selector_matches_selected_is_1_w += int(torch.equal(selector_w_bool, mask_w))
        selector_matches_selected_is_0_w += int(torch.equal(selector_w_bool, ~mask_w))
        selector_matches_selected_is_1_t += int(torch.equal(selector_t_bool, mask_t))
        selector_matches_selected_is_0_t += int(torch.equal(selector_t_bool, ~mask_t))

    print("---- summary ----")
    print(f"xnor_popcount_mismatch_across_seeds={xnor_popcount_mismatch}")
    print(f"total_query_rows={total_rows}")
    print(f"tie_sensitive_rows={tie_rows}")
    print(f"non_tie_rows={non_tie_rows}")
    print(f"non_tie_mismatch_rows={non_tie_mismatch_rows}")

    print("---- selector mask semantics ----")
    print(f"winner-tree: match(selected=1) in {selector_matches_selected_is_1_w}/{seeds} seeds")
    print(f"winner-tree: match(selected=0) in {selector_matches_selected_is_0_w}/{seeds} seeds")
    print(f"torch-topk : match(selected=1) in {selector_matches_selected_is_1_t}/{seeds} seeds")
    print(f"torch-topk : match(selected=0) in {selector_matches_selected_is_0_t}/{seeds} seeds")

    if non_tie_mismatch_rows == 0 and xnor_popcount_mismatch == 0:
        print("result: non-tie rows are equivalent between winner-tree and torch-topk")
        return 0

    print("result: found mismatches on non-tie rows or score computation")
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Check winner-tree vs torch-topk behavior")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--rows", type=int, default=17)
    parser.add_argument("--bits", type=int, default=64)
    parser.add_argument("--topk", type=int, default=7)
    parser.add_argument("--seeds", type=int, default=100)
    args = parser.parse_args()

    code = run_check(args.batch, args.rows, args.bits, args.topk, args.seeds)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
