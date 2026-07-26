# Learnable-logic research registry

This directory is the decision surface for new methods. It separates source
availability from experimental evidence and from deployment compliance.

## Files

- `branches.csv`: every public branch inspected in the two source repositories.
- `methods.csv`: one row per method family or materially different architecture.
- `results.csv`: selected, provenance-linked measurements. A low gap is never
  treated as a win without useful hard accuracy. `hard_acc` is the metric on
  `selection_split`; independent test measurements use the explicit `test_*`
  columns and cannot drive promotion.
- `synthesis.csv`: source-result-linked Boolean synthesis measurements,
  including mapped gates/depth/payload, runtime, artifact hashes, export replay,
  and formal equivalence status.
- `capacity.csv`: model-wide capacity and information-bottleneck measurements.
- `protocols.json`: matched experiment contracts used by result rows.
- `promotion_policy.json`: minimum evidence required to change the mainline.
- `validate.py`: structural, provenance, no-real-value, and promotion checks.
- `promotion.py`: paired candidate-versus-baseline decision logic.
- `render_method_table.py`: deterministic Markdown table renderer.

`METHOD_TABLE.md` is generated from the CSV/JSON files and is the concise human
view. The CSV/JSON files remain authoritative.

## Status vocabulary

- `reference`: current admissible baseline. This does not mean globally best.
- `candidate`: implemented and awaiting matched evidence.
- `screen`: bounded evidence only; not eligible for a headline claim.
- `rejected`: tested intervention that did not satisfy its promotion target.
- `external_ceiling`: useful accuracy reference with a noncompliant runtime.
- `unvalidated`: implementation exists but has no current compatible result.

## Non-negotiable deployment boundary

Training may use floating shadow parameters, optimizers, and STE surrogates.
The exported hard model must accept integer/Boolean input, carry only integer
or Boolean tensors and integer exponent metadata, and emit integer logits.
Binary values stored in floating tensors do not satisfy this rule.

Only `operator_audited_bool_int` is accepted for mainline promotion. A
bit-exact comparison against a hard floating carrier is useful evidence but is
not a complete no-real-value audit.

Promotion also requires a hashed standalone deployment payload. A training
checkpoint with floating shadow logits is not that payload, even when its
forward values happen to be binary.

`validate.py` recomputes every populated protocol hash from the authoritative
registry. Any result labeled `reference` or `promote` must also have a healthy
training status, strict runtime, zero floating tensors, a nonzero audit count,
and a standalone deployment payload hash. Older evidence that lacks this full
chain remains a baseline or screen rather than a headline result.

## Adding a method

1. Add a method row and an immutable protocol before running a comparison.
2. Add a capacity row, including the earliest information bottleneck.
3. Run the matched baseline and candidate with the same data, split, seeds,
   budget, selection rule, and architecture except for the declared variable.
4. Record the complete curve, best and final validation metrics, hard test
   result, runtime, model capacity, and hard-runtime audit.
5. Run `python -m research_registry.validate` and the repository tests.
6. Use `python -m research_registry.promotion --candidate ... --baseline ...`
   before changing a method to `reference`.

When one protocol contains multiple registered capacity variants, pass
`--candidate-variant` and `--baseline-variant` tokens that uniquely select the
corresponding result IDs. This keeps seeds paired within a capacity comparison
instead of silently pooling different model sizes.

No result may be promoted from a test-selected checkpoint, a single seed, an
unmatched budget, a partial runtime audit, or a non-finite/late-collapse run.
The current gate also requires at least `0.20 pp` mean paired validation hard
accuracy improvement; a merely positive sub-noise delta is retained as a
screen rather than called a method gain.
