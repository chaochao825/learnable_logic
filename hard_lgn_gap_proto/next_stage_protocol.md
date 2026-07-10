# Next-Stage Hard-LGN Protocol

This protocol follows the pasted reference text and reframes the project around
deployable hard-path accuracy, not only relaxed-full soft/hard gap.

Primary objective:

`maximize final hard-path accuracy under fixed gate budget, training budget, and seed list`

The three research axes must be evaluated separately before combining them.

## A. Random Wiring Search Then Freeze

Question:

Can random-connectivity search recover accuracy lost to poor initial wiring
while preserving a final hardware-simple fixed-connectivity LGN?

Comparison:

| method | connectivity | selection | final training |
|---|---|---|---|
| `fixed_random` | one random wiring | none | train gates on fixed wiring |
| `fixed_random_matched_budget` | one random wiring | none | train gates with epoch budget matched to search cost |
| `random_wiring_search_freeze` | many candidate wirings | validation hard accuracy | freeze selected wiring, retrain gates |
| `learnable_connectivity` | learned or candidate-refresh wiring | validation hard accuracy | optional, only if reliable existing code is available |

Validity rules:

- Separate `wiring_seed` from gate-logit `init_seed`; otherwise wiring quality is
  confounded with lucky initialization.
- Use a validation split for selecting wiring; report held-out test hard
  accuracy only after selection.
- Report search time and final training time separately; total time is the
  fair wall-clock cost of search.
- Include a matched-budget fixed-random baseline when claiming benefit over
  single wiring under equal training budget.
- Select by validation hard accuracy first, then native gap, unused ratio, and
  fanout/cost as tie breakers.

Required table:

`method | dataset | seed | wiring_seed | hard_acc | soft_acc | full_gap | native_gap | unused_gate_ratio | gate_count | depth | fanout_max | train_time`

## B. Registers / State / Sequential Hard-LGN

Question:

Do explicit Boolean registers let an LGN solve stateful tasks that a feedforward
LGN cannot solve at the same gate budget?

Minimal synchronous model:

```text
state_{t+1}, output_t = hard_logic_block(concat(input_t, state_t))
```

Comparison:

| method | state | role |
|---|---|---|
| `feedforward_lgn` | none | control; should fail delayed/stateful tasks |
| `register_lgn` | Boolean registers | proposed hard sequential logic |
| `rnn_small` | learned continuous state | neural reference |
| `gru_small` | gated continuous state | stronger neural reference |

Initial tasks:

- delayed copy
- sequence parity
- temporal majority
- finite-state-machine recognition

Validity rules:

- Evaluate sequence-level accuracy, not only per-step token accuracy.
- Report `register_utilization`: fraction of state bits that change over the
  validation/test set.
- Keep sequence length and state-bit budget fixed per comparison.
- Do not claim language modeling; these are controlled stateful logic tasks.

Required table:

`method | task | state_bits | time_steps | hard_acc | soft_acc | sequence_acc | native_gap | full_gap | train_time | register_utilization | gate_count | depth`

## C. Gate Redundancy + Task-Aware Truth-Table Hardening

Question:

Can redundant gates act as an error-correcting ensemble and improve final hard
accuracy after task-aware hardening?

Comparison:

| method | redundancy | regularization | hardening |
|---|---|---|---|
| `no_redundancy` | 1x | none | argmax |
| `more_gates_only` | 2x/4x/8x | none | argmax |
| `redundant_regularized` | 2x/4x/8x | activity, diversity, dropout | argmax |
| `redundant_task_hardened` | 2x/4x/8x | activity, diversity, dropout | validation-selected hardening |
| `redundant_abc` | selected redundancy | same | optional ABC post-pass |

Hardening candidates:

- direct argmax
- best-of-N Gumbel hard samples
- exact truth table when support is small
- sampled truth table when support is large
- support-limited enumeration when estimated support size <= `k`
- optional ABC structural optimization

Validity rules:

- Hardening must be task-aware: choose candidates by downstream validation hard
  loss/accuracy first, then refit error and circuit cost.
- Report best-hard-accuracy checkpoint versus best-soft-loss checkpoint.
- Do not claim to solve Mind-the-Gap if `full_acc_gap` remains large. The
  valid claim is improved deployable hard-path accuracy and native consistency.

Candidate score:

```text
score = hard_val_loss
      + lambda_refit * block_refit_error
      + lambda_cost * gate_count_or_depth
      + lambda_unused * unused_gate_ratio
```

Required tables:

- unified hard-accuracy leaderboard
- redundancy scaling curve
- hardening method comparison
- soft-hard gap taxonomy with `full_acc_gap`, `native_acc_gap`, and
  `block_refit_error`

## Reporting Discipline

- `full_acc_gap`: relaxed-full network vs hard-full network. Use for
  Mind-the-Gap comparison.
- `native_acc_gap`: method-native soft/path output vs final hard path. Use for
  block-wise or register-LGN method claims.
- `block_refit_error`: local relaxed-block-to-fitted-hard mismatch. Use for
  locating failed hardening stages.
- `hard_acc`: final deployable hard inference accuracy. This is the primary
  objective for the next stage.

Do not combine these metrics into a single success statement. A method can
improve `native_acc_gap` and `hard_acc` while still failing the original
relaxed-full Mind-the-Gap metric.
