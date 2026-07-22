# Server-210 Hard-LGN result capture

This directory preserves the unpublished 2026-07-01 Full-K propagation and
soft-loss precision sweeps from
`/home/spco/sow_linear/hard_lgn_gap_proto/runs` on server 210.  The matching
source scripts already in `hard_lgn_gap_proto/` have identical normalized
SHA256 values to the server copies:

| source | normalized SHA256 |
| --- | --- |
| `hard_lgn_benchmark.py` | `f11d7a764394bd93a72f3c3fc87dfde5a9877d749559acbf31ca0e03ce6c6a39` |
| `lightlogic_full_k_expansion.py` | `fcccf2629fd7069e6edabf7be77d0b784ff5318a2e62185c4b3089628f8b9220` |
| `soft_loss_precision_sweep.py` | `63937b204357ea61e4c7e421ea08a33c0207fa509cd56c9eb85a1a10e8c6bdd8` |

The main evidence is:

- Full popcount propagation closes the teacher gap to `0.1%` on the captured
  binarized-MNIST run at `K=16`, while thresholding every layer loses `9.6%`.
- On the small CIFAR-10 run, full popcount is within `0.5-0.9%` of the teacher;
  per-layer thresholding loses about `14.3-14.6%`.
- A float `concat_mlp` head reaches `96.43%` hard accuracy on the MNIST sweep,
  but is not a fully discrete result.  GroupSum rows retain a roughly `59-65%`
  gap, exposing the unresolved discrete representation/readout problem.

`full_k_boolean_digits_20260701` contains only a partial CSV and is explicitly
retained as an incomplete run, not counted as final evidence.  All files under
`raw/` are byte-preserved; `source_manifest.csv` records their SHA256 values.
