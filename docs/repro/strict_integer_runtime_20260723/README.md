# Strict no-real-value runtime evidence

Date: 2026-07-23

This bundle verifies two independent hard inference paths after the deployment
constraint was tightened: the Full-Discrete Wmag/A8 model and the original
two-input Hard-LGN prototype. The guarantee applies to the deployed model
forward. Continuous shadows, optimizers, STE/Gumbel gradients, refit losses,
and cross-entropy reporting remain training or measurement code.

## Full-Discrete checkpoint replay

Schema v6 exports integer weight codes, integer scale exponents, integer LUTs,
Boolean routing constants, and no floating inference state.
`StrictIntegerExecutor` accepts only `torch.uint8` images and carries every
activation as an `int64` code plus an `int32` exponent. Its runtime audit
inspects every Torch operator input and output.

| checkpoint | evaluated rows | strict accuracy | audited ops | real tensors | carrier comparison |
| --- | ---: | ---: | ---: | ---: | --- |
| d6/e192/h6, 50k | 5,000 | 75.30% | 14,319,802 | 0 | 100/100 exact logit rows, max diff 0 |
| d12/e384/h12/local4, 50k | 100 | 72.00% | 614,916 | 0 | 20/20 exact logit rows, max diff 0 |

The d6 result is the complete fixed 5,000-image validation split and exactly
reproduces the stored 75.30% final accuracy. The local4 row is a fixed-prefix
sanity replay, not a replacement for its stored full-validation result.

`int_matmul` is an integer-only CPU acceleration for fixed Wmag coefficients.
The unit suite also requires it to match the primitive A8-by-U4 LUT/shift-add
backend exactly. Neither backend creates real-valued model state.

## Hard-LGN Boolean replay

`StrictBooleanLogicExecutor` accepts only `torch.bool`, applies frozen integer
wiring and one of the 16 Boolean truth tables, and returns `int64` class
counts. GroupSum temperature division is removed from the decision path because
positive scaling cannot change count argmax. Cross entropy is computed only
after the executor returns, as an external research metric.

The smoke artifact contains 12 rows: four methods (`dlgn`, `gumbel_st`,
`block_relaxed`, and `block_hard_refit`) on three Boolean datasets. Every row
has `hard_runtime_domain=bool_int`, a positive audit operation count, and
`hard_runtime_floating_tensor_count=0`. These short runs validate execution,
not comparative accuracy.

The depth-only d12/e192 checkpoint has since completed the same full
5,000-image strict replay at `76.10%`, with zero floating tensors and 20/20
exact carrier-logit rows. Its payload hash, capacity audit, runtime sweep, and
training comparison are frozen in
[`../full_discrete_d12_strict_20260723/`](../full_discrete_d12_strict_20260723/README.md).

## Verification gates

- Full-Discrete module suite: 140/140 tests pass on the 236 H200 host's `lgn`
  environment.
- Hard-LGN strict executor plus benchmark tests: 11/11 pass in the same
  environment.
- Static AST tests reject float literals, true division, and floating
  activation/transcendental calls in both strict executor source files.
- Dynamic dispatch audits raise on any floating or complex tensor at any
  executed Torch operator.
- Unsupported Full-Discrete enhancement topology fails closed rather than
  falling back to the QAT floating carrier.

This is transaction-level proof, not cycle-accurate RTL. Fixed finite exponent
bounds, packed kernels, synthesis, timing, area, and energy remain separate
hardware deliverables.

## Files

- `fd_control_seed42_strict_integer_validation5000.json`: complete d6 replay.
- `fd_control_seed42_integer_vs_carrier_limit100.json`: exact d6 logit check.
- `scalelogic_local4_strict_integer_limit100.json`: local4 strict replay.
- `scalelogic_local4_integer_vs_carrier_limit20.json`: exact local4 logit check.
- `hard_lgn_strict_bool_smoke_results.csv`: per-method Boolean runtime fields.
- `hard_lgn_strict_bool_smoke_summary.md`: generated smoke summary.
- `validation.json`: frozen assertions and SHA-256 manifest.
