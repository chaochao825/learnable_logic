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
test accuracy. Causal routing lowers unused gates from 2.21% to 1.68% but is
still below the 38.21% integer trigram validation reference. The 100,000-window
v32-d2 and v64-d2 runs are therefore justified and queued; deeper models remain
conditional on the width result.

## Scaling path

1. Establish that causal routing beats the matched mixed candidate pool in
   validation hard accuracy, not only in soft accuracy or gap.
2. Double votes at fixed depth. Width advances only when hard validation
   accuracy and train hard accuracy both behave normally.
3. Add depth at fixed width. Reject depth when it increases mismatch, identity
   carry, unused gates, or context-support collapse without hard gain.
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

This phase is conditional. Building Boolean attention before the rolling model
beats integer n-gram references would add large fixed logic without evidence
that the learned LUT basis can fit language statistics.

## Promotion and stop rules

A configuration advances only when it improves validation hard accuracy under
the same data, budget, and seed. Gap reduction with collapsed accuracy is a
failure. Every accepted run must also show finite training, exact strict replay,
zero real-valued payload/runtime tensors, zero learned dense numeric matrices,
payload hashes, gate/depth/fanout/unused metrics, and reproducible split hashes.

Stop the current language-model branch if width does not improve hard accuracy,
if it remains below the integer bigram/trigram references after the full v64
screen, or if depth primarily increases identity paths and inactive vote planes.
In that case, the next justified experiment is hierarchical output coding or a
task-margin-aware hard refit, not a larger imitation Transformer.
