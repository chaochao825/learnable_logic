# CIFAR-100 strict LUT scale screen: partial evidence

Status: three of five pre-registered full configurations are complete. The
v128-d4 and H200 v256-d4 runs remain active. This partial table is deliberately
not imported into the unified result registry; the importer requires all five
rows.

Every completed row uses raw A8 input bit-planes, learned 4-input Boolean LUT
truth tables and discrete source indices, and fixed integer GroupSum. Full
validation/test execution and the 45,000-row training replay contain zero
floating tensors and no learned dense numeric matrix.

| variant | soft val | hard val | gap | train hard | test hard | time | unused | gates | depth | fanout | logical payload | mean class support | zero-support classes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flat-v64-d2 | 15.40% | 15.40% | 0.00 pp | 16.57% | 15.52% | 3,407.2s | 7.20% | 12,800 | 2 | 12 | 1,482,240 bits | 515.0 | 0 |
| spatial-v64-d2 | 16.94% | 17.08% | 0.14 pp | 18.17% | 16.94% | 4,976.4s | 6.65% | 12,800 | 2 | 13 | 1,482,240 bits | 282.6 | 0 |
| spatial-v128-d2 | 20.18% | 19.82% | 0.36 pp | 22.21% | 20.37% | 8,823.5s | 7.36% | 25,600 | 2 | 14 | 2,698,240 bits | 597.7 | 0 |

At matched depth, spatial routing improves hard validation accuracy by 1.68
percentage points over flat routing. Doubling spatial vote width then adds
2.74 points over v64. The second gain is accompanied by 2.12x mean class input
support, a 0.71-point increase in unused gates, and a 1.82x logical payload.
These are seed-0 capacity-screen results, not a promotion claim.

Evidence for each row includes the hard payload, payload and source hashes,
per-epoch and per-block metrics, structural diagnostics, and strict training
replay. See the frozen protocol at
[`docs/protocols/bitplane_lut_cifar100_scale_v1_20260724.md`](../../protocols/bitplane_lut_cifar100_scale_v1_20260724.md).
