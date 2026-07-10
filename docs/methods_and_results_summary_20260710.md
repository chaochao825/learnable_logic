# Methods And Results Summary

## 1. Scope

This summary consolidates the main learnable-logic experiments that were run in
`/home/spco/sow_linear/hard_lgn_gap_proto` on the 210 server. The work evolved
in stages:

1. LightLogic 2-input gate baselines and training-path variants
2. K-expansion from continuous gates to larger hard truth tables
3. Post-expansion threshold tuning and teacher-logit distillation
4. Direct `b>2` LUT training
5. Local Boolean minimization and XAG-style reporting

The goal of this publish tree is to preserve the method progression and the
best verified results without pushing the full raw run directory.

## 2. Experimental progression

### 2.1 Goal 0-2: LightLogic baseline, annealing, and ST/Gumbel-ST

Representative configurations:

- Boolean tasks (`parity8`, `majority9`, `random_sparse10`):
  `width=128`, `layers=4`, `epochs=120`, `batch=256`, `seed=0`
- Digits:
  `width=320`, `layers=4`, `epochs=150`, `train/test=4000/1000`, `seed=0`
- Methods:
  `light_iwp`, `light_iwp_anneal`, `light_iwp_st`, `light_iwp_gumbel_st`,
  `dlgn_op`

What this stage established:

- `light_iwp` is fully wired into the same fixed-topology evaluation flow as
  OP/DLGN.
- For 2-input gates, IWP uses 4 truth-table parameters per gate versus 16 for
  OP/DLGN, so the parameter ratio is exactly `0.25`.
- Goal 1 and Goal 2 were only partially supported. Annealing reduced gaps in
  some cases but could collapse gate utilization. ST/Gumbel-ST sometimes
  improved hard accuracy, but not uniformly across tasks.

Best visible rows from the merged report:

| dataset | best row | discrete_acc | note |
|---|---|---:|---|
| `majority9` | `light_iwp_st` | `1.0` | strongest LightLogic-family Boolean row in the initial sweep |
| `random_sparse10` | `light_iwp_st` | `0.9453125` | competitive with OP but not uniformly better |
| `digits` | `light_iwp_st` | `0.8955556` | below OP `0.9022222`, but close with 0.25x gate parameters |

Bottom line:

- Goal 0 baseline support is solid.
- Goal 1 and Goal 2 should be treated as conditional, not universal wins.

### 2.2 Goal 3-4: K-expansion and calibration

Representative configuration:

- Datasets: `parity8`, `majority9`, `random_sparse10`
- `K=2,4,8,16,32`
- Calibration modes: `fixed`, `per_layer`, `per_neuron`
- `width=128`, `layers=4`, `epochs=120`, `seed=0`

What this stage established:

- K-expansion is implemented end to end.
- Local truth-table approximation error (`local_mae`) improves monotonically as
  `K` grows.
- Network accuracy after hard thresholding does not improve monotonically with
  `K`.
- Calibration helps sometimes, but only partially: `4/30` same-`K`
  calibration comparisons beat the fixed-threshold baseline.

Best rows:

| dataset | best K/mode | expanded_acc | teacher_acc |
|---|---|---:|---:|
| `majority9` | `K=2`, `per_neuron` | `0.734375` | `1.0` |
| `parity8` | `K=2`, `fixed` | `0.5` | `0.21875` |
| `random_sparse10` | `K=4`, `fixed` | `0.859375` | `1.0` |

Bottom line:

- Larger local truth tables alone do not solve the hard deployment gap.
- Calibration is useful but unreliable under the current one-bit-per-layer
  thresholding scheme.

### 2.3 Goal 5-6: Distilled K-expanded hard students

Representative configuration:

- Full sweep mainly on `majority9` and `random_sparse10`, seed 0
- Confirmation runs on seeds 1 and 2 for `K=4`, `fixed`
- Distillation grid included `alpha in {0, 0.5, 2}` and `tau in {1, 2}`
- Teacher: continuous LightLogic
- Student: hard K-expanded network with learned thresholds

What this stage established:

- Goal 6 is meaningfully supported.
- Best distilled rows improved hard accuracy in `11/12` compared groups.
- KL distillation beat CE-only tuning in `10/12` groups.
- Goal 5, as originally formulated through independent weighted rounding, was
  degenerate: the weighted rounding changed ratio stayed `0`.

Best rows by dataset/seed:

| dataset | seed | K | init | distilled_hard_acc | delta_vs_initial |
|---|---:|---:|---|---:|---:|
| `majority9` | `0` | `4` | `fixed` | `0.8125` | `+0.21875` |
| `majority9` | `1` | `4` | `fixed` | `0.734375` | `+0.21875` |
| `majority9` | `2` | `4` | `fixed` | `0.859375` | `+0.2578125` |
| `random_sparse10` | `0` | `4` | `fixed` | `0.9921875` | `+0.1328125` |
| `random_sparse10` | `1` | `4` | `fixed` | `0.97265625` | `+0.2109375` |
| `random_sparse10` | `2` | `4` | `fixed` | `0.9921875` | `+0.05859375` |

Bottom line:

- Post-expansion tuning is the first stage that reliably moves hard deployed
  accuracy upward after K-expansion.
- The current benefit comes from threshold adaptation and teacher guidance, not
  from data-weighted rounding.

### 2.4 Goal 7: direct `b>2` LUT training

Representative direct-LUT configurations:

- Boolean:
  `b in {3,4}`, `width=160`, `layers=4`, `epochs=60`, `train/test=4000/1000`,
  `seeds=0,1,2`
- Digits:
  `b in {3,4}`, `width=240`, `layers=4`, `epochs=60`, `train/test=4000/1000`,
  `seeds=0,1,2`
- Matched binarized-MNIST:
  `b in {3,4}`, `width=800`, `layers=3`, `epochs=80`,
  `train/test=12000/3000`

This stage closes the most important old evidence gap: there is now real
end-to-end training evidence for local `b>2` LUT networks, not only
post-hoc decomposition of 2-input networks.

Best direct rows:

| dataset | best direct config | discrete_acc |
|---|---|---:|
| `binarized_mnist` | `b=3`, `width=800`, `layers=3`, `epochs=80`, `seed=0` | `0.8356667` |
| `digits` | `b=4`, `width=240`, `layers=4`, `epochs=60`, `seed=1` | `0.9177778` |
| `majority9` | `b=4`, `width=160`, `layers=4`, `epochs=60`, `seed=0` | `0.90625` |
| `parity8` | `b=3`, `width=160`, `layers=4`, `epochs=60`, `seed=1` | `0.53125` |
| `random_sparse10` | `b=3`, `width=160`, `layers=4`, `epochs=60`, `seed=0` | `0.97265625` |

### 2.5 Goal 8: LUT minimization and cost reporting

Goal 8 does not just train the LUTs. It exports local truth tables, minimizes
them with ABC, checks equivalence, and compares cost proxies:

- raw LUT implementation proxy
- ABC AIG `and/lev`
- local ANF/XAG-style `AND/XOR/NOT` estimates

For the direct `b>2` report, the important baseline mapping is:

| dataset | imported Goal 8 baseline | baseline config |
|---|---|---|
| `binarized_mnist` | `mnist_fixed` | `k=4`, `width=800`, `layers=3`, `epochs=80`, `train/test=12000/3000`, `seed=0` |
| `digits` | `digits_fixed` | `k=4`, `width=320`, `layers=4`, `epochs=150`, `train/test=4000/1000`, `seed=1` |
| `majority9` | `bool_per_neuron` | `k=2`, `width=128`, `layers=4`, `epochs=120`, `train/test=4000/1000`, `seed=2` |
| `parity8` | `bool_fixed` | `k=8`, `width=128`, `layers=4`, `epochs=120`, `train/test=4000/1000`, `seed=1` |
| `random_sparse10` | `bool_per_layer` | `k=2`, `width=128`, `layers=4`, `epochs=120`, `train/test=4000/1000`, `seed=2` |

Direct `b>2` versus imported `b=2` Goal 8 baselines:

| dataset | best direct acc | baseline acc | delta_acc | direct ABC AND | baseline ABC AND |
|---|---:|---:|---:|---:|---:|
| `binarized_mnist` | `0.8356667` | `0.7866667` | `+0.049` | `4125` | `1613` |
| `digits` | `0.9177778` | `0.9222222` | `-0.0044444` | `2224` | `547` |
| `majority9` | `0.90625` | `0.7421875` | `+0.1640625` | `784` | `37` |
| `parity8` | `0.53125` | `0.53125` | `0` | `0` | `2` |
| `random_sparse10` | `0.97265625` | `0.9375` | `+0.03515625` | `468` | `100` |

Interpretation:

- The strongest success case is matched-setting binarized-MNIST: direct
  `b=3` beats the matched `b=2` baseline by `+0.049`.
- `majority9` and `random_sparse10` improve meaningfully.
- `parity8` does not improve.
- `digits` is slightly below the imported baseline, but the baseline uses a
  stronger configuration (`width=320`, `epochs=150`) than the direct-b run
  (`width=240`, `epochs=60`), so this is not a clean apples-to-apples loss.
- Logic cost rises sharply for the direct `b>2` winners, so the accuracy gain
  is not free.

The complementary XAG report is local, not global:

- local ANF/XAG decomposition is exact per LUT truth table
- `xag_local_*` counts do not claim whole-network cross-LUT optimization
- ABC `and/lev` remains the primary whole-network minimized cost proxy

## 3. Main conclusions

1. Learnable logic training works, but plain 2-input LightLogic alone does not
   close the hard deployment gap robustly.
2. K-expansion improves local approximation quality but does not guarantee
   better end-to-end hard accuracy.
3. Post-expansion threshold tuning and teacher-logit distillation are the most
   reliable interventions before moving to larger local LUTs.
4. Direct `b>2` LUT training provides the strongest end-to-end evidence so far,
   especially on matched binarized-MNIST.
5. The accuracy gains from direct larger LUTs come with a real logic-cost
   increase, which makes Goal 8 minimization and future structural optimization
   necessary rather than optional.

## 4. Included supporting artifacts

The repository includes:

- source scripts from `hard_lgn_gap_proto`
- stage reports under `docs/reports/`
- selected summary CSVs under `docs/tables/`

The repository intentionally does not include the full raw `runs/` tree. Those
artifacts remain on the 210 server for deeper audit and reruns.
