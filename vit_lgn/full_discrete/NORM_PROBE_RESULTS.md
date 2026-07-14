# Frozen-checkpoint normalization probe

This probe is a counterfactual on fixed weights, not a substitute for
from-scratch training.

- checkpoint: `full_discrete_w8a8_d6e192_seed42`, step 50,000, seed 42
- checkpoint protocol: `3e1ba64ea4a7e53af3a3563e2536269a07a2939654f9510d4690398a7cbdc8c5`
- validation: the complete 5,000-example split selected with seed `20260711`
- model: depth 6, embedding 192, Wmag7/A8, Top-K 8
- all variants load the identical learned state with `strict=True`
- rerun after freezing Top-K ties as score-descending/key-index-ascending
- probed `model.py` SHA256:
  `4c882b9b4db914383b528f9656df487c8428be2a16ca103947d17d3e9d86fcae`

| Block norm | Final norm | Accuracy | Cross entropy |
|---|---|---:|---:|
| Q15 RMS LUT | same | 74.68% | 0.8689 |
| exact-nearest Shift-RMS | same | 73.60% | 0.9210 |
| requant only | same | 15.30% | 15.4844 |
| identity | same | 15.30% | 15.4844 |
| exact-nearest Shift-RMS | identity | 73.60% | 1.8611 |
| Q15 RMS LUT | identity | 74.54% | 1.7670 |

The exact-nearest Shift-RMS control uses the unrounded sum of squares and
constant cross-comparisons.  It improves on the earlier rounded-mean prototype
(72.98%) and is within 1.08 percentage points of Q15 RMS without retraining.

Deleting every block norm destroys this checkpoint.  Deleting only the final
norm nearly preserves argmax accuracy but substantially worsens calibration,
which is consistent with a bias-free classifier being approximately invariant
to a positive scalar while its logit magnitude is not.

Formal effectiveness decisions are deferred to `launch_logic_norm_queue_210.sh`;
each queued configuration runs 50,000 steps with a paired control and identical
post-construction data RNG reset.
