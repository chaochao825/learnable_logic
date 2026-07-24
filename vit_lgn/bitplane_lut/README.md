# A8 bit-plane Hard-LGN

This package is the logic-native successor candidate to the FullDiscrete A8
integer baseline.  A persistent state is represented as groups of eight
Boolean bit-planes.  Every learned deployment parameter is either a LUT truth
bit or a discrete source index.  Fixed input routing, GroupSum/popcount,
comparators, and integer counters are support logic; learned dense integer or
floating matrices are outside this method boundary.

The first registered experiment is deliberately small and falsifiable.  It
uses the 8x8 sklearn digits input as 64 unsigned A8 symbols, preserves all 512
raw planes through fixed bypass wires, and learns only the additional vote
planes with 4-input LUT blocks.  Each block is hardened/refitted and frozen
before the next block trains.  The strict executor accepts Boolean input planes
and returns integer class counts while a Torch dispatch audit rejects every
real-valued runtime tensor.

The initial v1 smoke diagnosed class-imbalanced truth refitting and is retained
as negative design evidence.  Full-data comparisons use the separately frozen
`docs/protocols/bitplane_lut_digits_scale_v2_20260724.md` contract.  Results
are not promoted from a smoke run or from a floating training carrier.

## Registered v2 result

Across three paired seeds, direct hard-ST/argmax reaches 88.02% validation and
85.31% test hard accuracy with 160 learned vote planes. Doubling only the
learned vote region to 320 planes reaches 91.48% validation and 89.75% test:
gains of 3.46 and 4.44 percentage points, with all three validation seeds
improving. All vote planes are active and all validation/test hard logits match
the strict Boolean/integer executor exactly.

Class-balanced independent truth refitting is not the winner. It lowers mean
validation hard accuracy by 1.67 percentage points versus argmax across both
widths. Greedy wiring/truth refit changes no additional source and reproduces
the truth-refit payloads. This is a rejection of the current local refit
objective, not of learned wiring: payload reconstruction shows that training
moves 65.27%-73.67% of gate inputs away from their default candidate route.

The complete 18-run table, curves, payload hashes, state diagnostics, and
post-hoc wiring audit are under
`docs/repro/bitplane_lut_digits_v2_20260724/`.

The same 18 payloads have also been exported and optimized with ABC. Every
source BLIF matches 257 deterministic random vectors and every optimized BLIF
passes ABC CEC. The learned functions are only weakly K=4 compressible:
76%-79% use all four truth-table inputs, dictionary coding increases table
storage by 17%-22%, and ABC reduces mapped LUT count by only 2.7%-6.9% while
raising mapped depth from two to three. The full synthesis evidence is under
`docs/repro/bitplane_lut_abc_20260724/`.

The 160- and 320-vote models contain 5,120 and 10,240 raw truth bits,
respectively (`2 blocks * votes * 16`). They still require discrete source
indices, so the synthesis audit uses 17,920 and 35,840 bits as the complete
learned LUT-plus-wiring payload before optimization.

## Reproduction entry points

Run one registered cell with, for example:

```bash
python -m vit_lgn.bitplane_lut.train_digits \
  --out-dir runs/v2_argmax_w672_s0 \
  --state-bits 672 --refit-mode argmax --seed 0
```

After completing the registered matrix, reconstruct learned candidate ranks
and regenerate the hash-checked tables:

```bash
python -m vit_lgn.bitplane_lut.analyze_payload_wiring \
  --runs-root runs --out runs/payload_wiring_diagnostics.csv
python vit_lgn/bitplane_lut/summarize_digits.py \
  --runs-root runs \
  --out-dir docs/repro/bitplane_lut_digits_v2_20260724 \
  --wiring-diagnostics runs/payload_wiring_diagnostics.csv \
  --source-root vit_lgn/bitplane_lut
```

Each run writes `per_epoch.csv`, `block_results.csv`, `result.json`, a source
and split-hashed `run_manifest.json`, and the standalone `hard_payload.pt`.

Use `synthesize_payloads.py export`, `abc`, and `summarize` to reproduce the
BLIF export, ABC optimization, formal equivalence checks, and synthesis tables.

## Strict character-model feasibility

The frozen Tiny Shakespeare protocol extends the same deployment boundary to
next-character prediction. A rolling 64-character context is encoded as 512
protected Boolean planes. The learned payload remains only 4-input truth bits
and discrete source indices, followed by fixed integer GroupSum; there is no
embedding, attention, MLP, or learned dense numeric matrix. `mixed` routing is
the matched global-candidate baseline and `sequence_causal` adds fixed recent
history and same-class vote candidates.

Run a pre-registered cell through:

```bash
vit_lgn/bitplane_lut/run_shakespeare_scale.sh \
  causal-v32-d2-s0 3 sequence_causal 32 2 20 6 4 100000 0
```

The protocol and capacity gates are recorded in
`docs/protocols/bitplane_lut_shakespeare_char_scale_v1_20260724.md` and the
machine-readable research registry. Seed 0 is only a feasibility screen;
promotion requires the matched seeds 1 and 2.

In the matched 20,000-window smoke, causal routing improves validation hard
accuracy from 25.82% to 28.99% and test hard accuracy from 25.59% to 28.50%
at the same 4,160-gate budget. Unused gates fall from 2.21% to 1.68%, while the
accuracy gap increases slightly from 2.74% to 3.08%. The candidate is still
below the 38.21% integer trigram reference, so this result authorizes the full
width ladder but is not a language-model promotion.
