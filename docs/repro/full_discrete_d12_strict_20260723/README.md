# FullDiscrete d12/e192 strict replay

## Decision

The existing `full_discrete_a8/d12e192` checkpoint becomes the strongest
admissible no-real-value accuracy reference in this repository. Its standalone
schema-v6 executor obtains `76.10%` on all 5,000 fixed CIFAR-10 validation
images, exactly reproducing the stored final checkpoint accuracy. The runtime
audit observes zero floating or complex tensors.

This does **not** promote a new method and does not establish a robust depth
scaling law. It is one seed of the same FullDiscrete method. Relative to the
matched d6/e192 training run, d12/e192 improves final validation accuracy by
`0.80 pp` and best validation accuracy by `0.96 pp`, while using `1.993x`
parameters and `1.706x` training wall time. Both first exceed 75% at 35k steps;
only d12 exceeds 76%, at 40k.

## No-real-value boundary

The deployment artifact and strict executor satisfy the repository's strong
runtime definition:

- input is `uint8` and persistent activations are Boolean/integer codes plus
  integer exponents;
- output logits are integer code/exponent pairs;
- the standalone payload contains only `uint8`, `int16`, `int32`, and `int64`
  tensors;
- all 1,960,214 audited Torch dispatches in the full replay observe zero real
  or complex tensors;
- 20/20 rows have logits exactly equal to the offline QAT hard carrier, with
  maximum absolute difference zero and no prediction mismatches.

The differentiable optimizer necessarily used floating shadow weights,
gradients, and loss. Those training values are not part of the deployment
payload. Calling differentiable training itself real-free would be incorrect;
the claim here is a completely Boolean/integer exported inference artifact.

| proof item | value |
| --- | --- |
| checkpoint SHA-256 | `02d22b92d520a3c7e18f97186a0d594a804aeabcf6168e022d4730b949db4181` |
| deployment payload SHA-256 | `51ccd7bbfab9877c9cefdc070d836620be0fbc2b985fb517c7a7ada16f28a60a` |
| payload serialized size | 85,381,242 bytes |
| strict rows / correct | 5,000 / 3,805 |
| strict accuracy | **76.10%** |
| exact carrier rows | 20/20 |
| max logit difference | 0 |
| runtime floating tensors | **0** |

## Matched training result

Both runs use the same CIFAR archive, seed 42, fixed 5,000-row validation
split, dimension 192, six heads, A8 state, Wmag7 projections, Top-K 8, batch
128, 50k steps, 2k warm-up, AdamW settings, and source hashes. Depth is the
declared variable.

| config | parameters | train time | best validation | best step | final validation | first 75% | first 76% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| d6/e192 | 3,562,752 | 8,699.6 s | 75.34% | 40k | 75.30% | 35k | never |
| d12/e192 | 7,101,696 | 14,838.2 s | **76.30%** | 40k | **76.10%** | 35k | 40k |

All ten stored validation points and training losses are finite. Both models
finish the full 50k budget. The d12 final result is 0.20 pp below its selected
40k best point, so there is no collapse but also no evidence that extra depth
improves convergence speed. Three paired seeds are required before a depth
scaling claim can pass the promotion gate.

## Capacity and cost

| deployed capacity | d6/e192 | d12/e192 | ratio |
| --- | ---: | ---: | ---: |
| transformer blocks | 6 | 12 | 2.000x |
| shift-add layers | 32 | 62 | 1.938x |
| logical weight coefficients | 3,550,080 | 7,089,024 | 1.997x |
| logical Wmag7+sign bits | 28,400,640 | 56,712,192 | 1.997x |
| audit-payload tensor bits | 341,840,272 | 681,822,688 | 1.995x |
| dense fan-in maximum | 768 | 768 | 1.000x |
| dense nonzero fanout maximum | 768 | 768 | 1.000x |
| general learned matrix multipliers | 0 | 0 | - |

The 81.28 MiB d12 tensor inventory is an **audit payload**, not a compact
hardware image: schema v6 intentionally stores redundant code, sign, unpacked
bit-plane, and chunk forms. A canonical bit-packed weight representation is
6.76 MiB before wiring and fixed arithmetic. Neither number is a synthesized
gate count.

`12` is transformer-block depth, not Boolean critical-path depth. Exact gate
count, full logic depth, fanout after buffering, and inactive/unused gate ratio
remain unavailable until bit-level lowering and ABC/Yosys synthesis. They are
left blank in the result registry instead of being inferred from parameter
count. This distinction prevents the capacity accounting error present in
earlier informal experiments.

## Runtime

The full d12 replay ran on server 434, an Intel Xeon Gold 6348 CPU, with Python
3.8.10, PyTorch 2.4.1, batch 64, 48 Torch threads, and the `int_matmul`
transaction backend.

| run | rows | accuracy | elapsed | throughput | floating tensors |
| --- | ---: | ---: | ---: | ---: | ---: |
| d12 full validation | 5,000 | 76.10% | 1,754.7 s | 2.850 image/s | 0 |
| d6 matched-host prefix | 100 | 78.00% | 16.87 s | 5.926 image/s | 0 |
| d12 matched-host prefix | 100 | 83.00% | 33.75 s | 2.963 image/s | 0 |

The matched 100-row prefixes show the expected near-2x depth cost. Their
accuracies are prefix diagnostics, not model comparisons. Dispatch audit counts
change with batching and prove dtype coverage only; they are not operation or
gate-count estimates. The Python transaction executor is also not the claimed
million-image/s compiled Boolean inference path from DLGN work.

## Reproduction

The immutable training checkpoint remains at:

```text
/home/wangmeiqi/learnable_logic_full_discrete_scale_ablation_20260713/runs/fdscale_d12e192_seed42/checkpoint.pt
```

The isolated replay workspace is:

```text
/home/wangmeiqi/learnable_logic_d12e192_strict_audit_20260723
```

From that workspace on server 434, the full payload replay is equivalent to:

```bash
/home/wangmeiqi/miniconda3/envs/lgn/bin/python -m \
  vit_lgn.full_discrete.evaluate_integer_checkpoint \
  --payload remote_runs/d12e192_strict/deployment_payload.pt \
  --data-root /home/wangmeiqi/ViT-LGN_goal6plus_nobias_retry_20260630/data/cifar-10 \
  --split validation --valid-size 5000 --batch-size 64 --threads 48 \
  --linear-backend int_matmul --output strict_integer_validation5000.json
```

Run `python freeze_evidence.py` in this evidence directory to recheck the
protocol, results, hashes, no-real assertions, exact logits, and capacity
relationships. The canonical strict protocol SHA-256 is
`72c4f07bbdd533906e0b026d213ecb20768de65ddcf087a1a5fc8d4fec05bd53`.

## Mainline consequence

No wider model or new one-bit mechanism should be launched from this result.
The controlled sequence is:

1. compact the redundant payload and prove exact equivalence;
2. confirm d6 versus d12 with three paired seeds before claiming depth scaling;
3. test clipping, augmentation, or useful learning-rate duration one variable
   at a time;
4. lower the retained A8/Wmag network to bit-sliced logic and measure real gate
   count, logic depth, fanout, equivalence, synthesis runtime, and PPA.

Block-wise truth-table refitting remains useful only inside an accurate
multi-bit carrier or for bounded subblocks. Repeating whole-network one-bit
width expansion would ignore the demonstrated hard-capacity bottleneck.

## Files

- `protocol.json`: strict replay protocol used by the research registry.
- `comparison.csv` and `comparison.json`: generated d6/d12 training, runtime,
  and capacity comparison.
- `raw/training/`: matched source training protocols and complete curves.
- `raw/d12/`: full replay, exact-logit check, environment, tests, payload
  capacity, benchmark sweep, and remote run manifest.
- `raw/d6/`: prior full strict replay, matched-host prefix runtime, and payload
  capacity control.
- `validation.json`: deterministic assertions and SHA-256 manifest.
- `freeze_evidence.py`: standard-library evidence checker.
