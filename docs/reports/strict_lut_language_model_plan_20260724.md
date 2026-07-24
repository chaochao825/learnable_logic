# Strict LUT language-model feasibility plan

## Decision

The current A8 bit-plane Hard-LGN can be tested as a small autoregressive
language model, but it is not a drop-in discrete implementation of microGPT or
nanoGPT. Their learned embeddings, dense Q/K/V and MLP projections, and
floating normalization would violate the method boundary. The first valid
question is narrower: can a Boolean LUT/wiring network improve next-character
hard accuracy as width and depth increase while preserving zero-real strict
execution?

The official [nanoGPT repository](https://github.com/karpathy/nanoGPT) and
[microgpt reference](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
remain noncompliant accuracy and optimization references. They are not mixed
into the logic-native model.

## Hard boundary

Allowed learned deployment state:

- Boolean 2/3/4-input LUT truth bits.
- Discrete source indices and discrete wiring.
- Boolean state bit-planes and hard-frozen block outputs.

Allowed fixed support logic:

- Token/character ID to bit-plane decomposition.
- Popcount, carry, comparison, argmax, and integer GroupSum.
- Fixed positional or causal routing.

Rejected deployment capacity:

- Floating embedding, attention, normalization, MLP, or classifier weights.
- Learned dense integer matrices or quantized GPT projections.
- A learned integer token table presented as an embedding rather than exported
  and accounted for as Boolean LUT/gate logic.

Training may retain real-valued shadows for hard-ST gradients. Those shadows
are never exported and do not participate in strict inference.

## Active phase: rolling character model

The frozen v1 protocol uses 64 previous Tiny Shakespeare characters, encoded
as 512 protected Boolean planes. It compares global mixed routing against
causal routing with recent-history and same-class vote candidates. The readout
is fixed integer GroupSum over 65 characters.

| configuration | gates | depth | logical payload |
| --- | ---: | ---: | ---: |
| mixed/causal v32-d2 | 4,160 | 2 | 304,128 bits |
| causal v64-d2 | 8,320 | 2 | 636,928 bits |
| causal v64-d4 | 16,640 | 4 | 1,202,688 bits |
| causal v128-d4 | 33,280 | 4 | 2,533,888 bits |

The first matched smoke uses 20,000 fixed windows and four epochs per block.
Only a routing winner advances to the 100,000-window width/depth ladder. Seed 0
is a screen; seed 1/2 repeats are mandatory before promotion.

The completed matched smoke selects causal routing: validation hard accuracy
is 28.99% versus 25.82% for mixed routing, with 28.50% versus 25.59% report-only
test accuracy. Causal routing lowers unused gates from 2.21% to 1.68%.

The 100,000-window fixed-depth width screen is also complete. Causal v32-d2
reaches 33.82% validation and 33.16% report-only test hard accuracy. Causal
v64-d2 reaches 36.275% validation and 35.434% test hard accuracy, gains 2.455
and 2.276 percentage points respectively, and raises training hard accuracy by
4.099 points. The final validation soft/hard gap is 0.14 points and strict
execution audits 63,216 operations with zero floating tensors. Width therefore
scales task accuracy, not only the gap. It also raises the unused-gate ratio
from 2.57% to 4.01%, so utilization remains an explicit optimization target.
The v32 payload has three zero-input-support output classes (`$`, `&`, and
`3`); the first two are absent from the sampled training windows, while `3`
exposes a learned class-support collapse. V64 reduces this count from three to
zero and raises mean class input support from 95.7 to 186.2 raw context bits.
The v64 result is still 1.940 points below the 38.215% integer trigram
reference.

Causal v64-d4 is complete at 39.935% validation, 39.403% report-only test, and
46.825% training hard accuracy. It improves 3.660 validation points over
v64-d2 and exceeds the integer trigram validation reference by 1.720 points.
Its final soft/hard accuracy gap is 0.005 points. The two prefix blocks are
tensor-identical to the d2 payload, remain frozen, and are charged their
original training time; only blocks three and four were newly trained. The
strict executor audits 113,808 operations with zero floating tensors, and the
payload contains zero learned numeric weights or dense integer matrices.

Depth doubles the logical payload from 636,928 to 1,202,688 bits, raises unused
gates from 4.01% to 4.60%, and raises mean class input support from 186.2 to
397.8 raw context bits. For context, the corpus trigram count table needs about
3.845 Mbit as a dense 14-bit counter table, while an ideal sparse-entry lower
bound is about 0.382 Mbit before lookup logic. The current result establishes a
hard-accuracy depth gain, not superior storage efficiency.

A support-only v128-d2 prefix runs in parallel on otherwise idle hardware. The
tracked conditional launcher starts the registered v128-d4 continuation only
if v128-d2 beats v64-d2 and its strict audit contains zero floating tensors or
numeric matrices. The completed v64-d4 already passes the other gate checks:
it beats both v64-d2 and trigram, and its exact prefix hash matches v64-d2.

## Scaling path

1. Establish that causal routing beats the matched mixed candidate pool in
   validation hard accuracy, not only in soft accuracy or gap.
2. Double votes at fixed depth. Width advances only when hard validation
   accuracy and train hard accuracy both behave normally.
3. Add depth at fixed width. Reject depth when it increases mismatch, identity
   carry, unused gates, or context-support collapse without hard gain.
   Deeper runs must load the shallower strict payload, verify exact logits, and
   train only the appended blocks.
4. Measure structural context support for every final vote and class. Protected
   input planes prevent destructive state loss, but they do not guarantee that
   learned functions actually use long-range context.
5. Replace flat 65-way GroupSum only after the direct character ladder scales.
   A Boolean radix/tree decoder is the preferred route for byte or subword
   vocabularies because flat votes scale as `vocabulary * votes_per_class`.

## Toward a Boolean GPT-like block

A later block can preserve the causal-token interface while replacing each GPT
component explicitly:

| GPT component | strict candidate | gate to proceed |
| --- | --- | --- |
| token embedding | raw token-ID bit-planes or synthesized Boolean code network | no learned dense table |
| Q/K/V projection | local LUT networks over state planes | hard accuracy scales with width |
| attention score | fixed XNOR/popcount plus integer comparator | exact Boolean/integer replay |
| attention routing | fixed or discrete learned Top-K wiring | fanout and depth reported |
| value aggregation | bit-sliced counters/carry logic | no dense integer projection |
| MLP | residual LUT blocks with protected planes | avoids identity and inactive collapse |
| output head | hierarchical Boolean code/tree | vocabulary cost below flat GroupSum |

This phase remains conditional. The seed-0 rolling model now beats trigram, but
building Boolean attention before seed repeats and the v128 scale point would
add large fixed logic without enough evidence that the gain is reproducible.

## Promotion and stop rules

A configuration advances only when it improves validation hard accuracy under
the same data, budget, and seed. Gap reduction with collapsed accuracy is a
failure. Every accepted run must also show finite training, exact strict replay,
zero real-valued payload/runtime tensors, zero learned dense numeric matrices,
payload hashes, gate/depth/fanout/unused metrics, and reproducible split hashes.

The v64-d4 depth gate passes: hard validation accuracy improves, exceeds
trigram, and does not show inactive or class-support collapse. Causal v128-d4
is now conditional only on the running v128-d2 support prefix beating v64-d2
and passing strict replay. Stop or redirect the branch if v128 regresses, if
seed-1/2 do not reproduce the gain, or if the larger vocabulary/output-tree
stage fails. The next fallback is hierarchical output coding or a
task-margin-aware hard refit, not a larger imitation Transformer.
