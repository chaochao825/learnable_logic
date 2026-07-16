# ScaleLogic/Hadamard review record

Date: 2026-07-16

## Scope and review route

The review covered the fixed Hadamard global mixer, mixed-mixer model wiring,
schema-v2 export, frozen training launchers, hardware claims and the scaling
report.  Two independent Codex subagent reviews were requested and both failed
before analysis with HTTP 429 retry exhaustion.  The local Claude and Gemini
CLI health checks then short-circuited because neither executable is installed.
Following the SSH development fallback rule, the review was completed with a
fresh isolated remote copy, adversarial local inspection and full tests.  The
formal training directory was never modified.

## Findings and dispositions

1. **Hybrid exporter topology lost the retained attention shape.**  When block
   zero was fixed Hadamard and a later periodic block retained hard attention,
   topology discovery searched only block zero and emitted zero heads/Top-K.
   It now searches the complete module graph, exports `head_dim`, and has a
   regression test whose first block is fixed and third block is attention.
2. **Hadamard was described too loosely as A8-compatible.**  Its input codes
   are A8, but the 64-token transforms require 14- and 20-bit accumulators and
   the `branch_shift=2` output needs 13 signed bits in the worst case.  Schema
   v2 now exports and validates every width plus the enclosing residual-A8
   requantization boundary.
3. **One schema-v2 validation error still said version 1.**  The diagnostic was
   corrected.
4. **The formal source must remain traceable after review hardening.**  Commit
   `54229c1` preserves the exact source set used by the active 50k run.  Later
   ABI metadata changes intentionally form a separate commit; they do not
   retroactively alter the training protocol.

No blocker was found in the hard Hadamard value path, deterministic mask,
periodic replacement wiring or fixed-source launcher.  The remaining material
limitation is empirical: all fixed-Hadamard variants underperformed hard
content routing at 1k, so they are negative ablations rather than the selected
accuracy path.

## Verification evidence

- Formal ordered source-set SHA256 at commit `54229c1`:
  `9e8af5d0bd8a42e3b3e913d6a94cbdb9f654c14f005c4db5e6697bb0c6997c94`.
- Formal `enhancements_hadamard.py` SHA256:
  `75a324549fb0429bc71f1c1fd639e499ee76adacddfc9be395973ab5af3ba904`.
- Active candidate protocol SHA256:
  `fe9b968f3be661a3ed8cbf77649978774ba525afddf725fe79c127523ec37198`.
- Isolated 434 review tree: 117/117 Python unit tests passed after the two
  findings above were fixed.
- `bash -n` passed for both ScaleLogic launchers; `compileall` passed for the
  complete package.
- `git diff --check` and the bounded credential/private-key scan passed.

The 5k candidate accuracy is only an intermediate observation.  Effectiveness
still requires the candidate and local0 control to finish 50,000 steps.
