## Research change

- Method ID:
- Baseline method ID:
- Matched protocol ID:
- Declared independent variable:

## Required checks

- [ ] The method, protocol, result, and capacity registries are updated.
- [ ] Candidate and baseline use identical data, split, seeds, budget, and selection.
- [ ] At least three paired seeds are reported for a promotion claim.
- [ ] Complete per-epoch curves are finite and late validation drop is within policy.
- [ ] Mean paired validation hard accuracy improves by at least 0.20 pp; a smaller soft/hard gap alone is not presented as a win.
- [ ] `hard_acc` is validation-selected; independent `test_*` fields did not select the method.
- [ ] Gate/state utilization and the earliest information bottleneck are reported.
- [ ] Exported inference accepts bool/int input and executes only bool/int tensors.
- [ ] A standalone hard payload is hashed; a floating training checkpoint is not submitted as deployment proof.
- [ ] Operator audit reports zero floating/complex tensors and a nonzero op count.
- [ ] Source commit, protocol hash, environment, and artifact hashes are recorded.
- [ ] `python -m research_registry.validate` passes.
- [ ] `python -m research_registry.render_method_table --check` passes.

## Decision

- [ ] Promote
- [ ] Continue as bounded screen
- [ ] Reject and retain as negative evidence
