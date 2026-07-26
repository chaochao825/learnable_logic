# Branch consolidation audit (2026-07-23)

## Complete public re-audit

The table below records the earlier merge transaction. A second fetch and
ancestry audit after that transaction found ten public branches in
`chaochao825/learnable_logic`. Current `main` and
`codex/integrate-bitstate-hard-lgn-20260723` both point to
`325f2498d85469a95bd82bf113c350f0890a9710`; every other public branch head is
an ancestor of that commit. No learnable-logic source branch remains to be
merged, and rerunning an old branch as though it were a new method is not
permitted. The authoritative machine-readable inventory is
[`../research_registry/branches.csv`](../research_registry/branches.csv).

`Vector-GitHub/ViT-LGN` has two public heads. `COMBINE` at `85dfd458...` and
`Width_Expansion` at `a9253762...` diverge from merge base `9fd8fe3...` and are
kept as external evidence, not merged blindly. COMBINE is a useful
hard-forward training baseline but retains a floating carrier. Width Expansion
widens only after an 18-bit patch bottleneck and has no matched strict result.
Their method dispositions and capacity bottlenecks are in
[`../research_registry/METHOD_TABLE.md`](../research_registry/METHOD_TABLE.md).

## Scope

The integration branch starts from `origin/main` at `8aa65fb` and merges
`origin/codex/logic-hadamard-mixer` at `cae858e` with an explicit merge commit.
Together those two histories cover every remote feature head that existed at
the start of the audit.

| Remote branch | Head | Disposition |
| --- | --- | --- |
| `main` | `8aa65fb` | First parent; retains the 20k ScaleLogic checkpoint audit. |
| `agent/add-full-discrete-scalelogic` | `516561f` | Already contained by `main`. |
| `codex/full-discrete-enhancements` | `47e0250` | Ancestor of the merged Hadamard branch. |
| `codex/full-discrete-results-20260715` | `2b61262` | Ancestor of the merged Hadamard branch. |
| `codex/full-discrete-scale` | `2984167` | Ancestor of the merged Hadamard branch. |
| `codex/logic-gate-backend` | `4523f31` | Ancestor of the merged Hadamard branch. |
| `codex/logic-hadamard-mixer` | `cae858e` | Merged as the second parent. |
| `codex/logic-tree-conv` | `376aba4` | Ancestor of the merged Hadamard branch. |
| `vit-lgn-merge-20260710-145556` | `398b041` | Ancestor of the merged Hadamard branch. |

The merge base of `main` and `codex/logic-hadamard-mixer` is
`39c73d099bdcd7068f99bdea7667ae54578c193c`.  `main` contributes two unique
commits and the Hadamard branch contributes 19 unique commits.

## Conflict policy

Thirteen files conflicted.  For implementation and full-discrete experiment
files, the Hadamard side was selected because it contains the cumulative
export schema, tests, launchers, and global-LUT negative result.  The root
README was then amended with `main`'s later 5k/10k/15k/20k ScaleLogic evidence
and links to its protocol and source-hash records.

The conflicted paths were:

- `README.md`
- `vit_lgn/full_discrete/NORM_PROBE_RESULTS.md`
- `vit_lgn/full_discrete/README.md`
- `vit_lgn/full_discrete/__init__.py`
- `vit_lgn/full_discrete/enhanced_model.py`
- `vit_lgn/full_discrete/enhancements_hadamard.py`
- `vit_lgn/full_discrete/export_logic_payload.py`
- `vit_lgn/full_discrete/launch_scalelogic_50k_434.sh`
- `vit_lgn/full_discrete/launch_scalelogic_pair_queue_434.sh`
- `vit_lgn/full_discrete/test_enhancements_hadamard.py`
- `vit_lgn/full_discrete/test_export_logic_payload.py`
- `vit_lgn/full_discrete/test_train_protocol.py`
- `vit_lgn/full_discrete/train_cifar.py`

## Uncommitted 210 snapshot

The active server-210 directory
`/home/spco/sow_linear/learnable_logic_hadamard_mixer_20260716` was intentionally
left untouched.  Its seven changed or untracked files were copied byte-for-byte
to [`docs/repro/210_active_hadamard_snapshot_20260723/`](repro/210_active_hadamard_snapshot_20260723/)
for provenance only.  They are not silently overlaid onto the merged source;
their behavior must be reviewed and tested before selective integration.
