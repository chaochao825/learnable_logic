# Bit-plane LUT Shakespeare character-model scale protocol v1

Status: frozen before any Shakespeare Hard-LGN training result is inspected.

## Question

Test whether the logic-native A8 bit-plane LUT method can learn a rolling-window
character language model and improve with Boolean vote width/depth. This is a
strict next-character microGPT feasibility screen, not a claim that the model
implements Transformer attention.

## Data

- Source: Karpathy `char-rnn` Tiny Shakespeare `input.txt`.
- File bytes: 1,115,394.
- SHA-256: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.
- Vocabulary: the sorted 65 unique characters, encoded as integer IDs 0-64.
- Split: first 90% train, next 5% validation, final 5% test. Context windows
  never cross a split boundary.
- Context: 64 previous character IDs as 64 A8 symbols and exactly 512 Boolean
  bit-planes; label is the next character.
- Training screen: 100,000 fixed sampled train windows. Validation selection:
  20,000 fixed sampled windows. Final test: every valid window in the held-out
  final 5%. Sampling is uniform without replacement with seed 20260724.
- The 65-class output is never reduced when a sampled split omits a rare
  character. Missing characters are recorded in the run manifest and remain
  all-negative output classes; class-weight denominators are clamped to one.

## Method boundary

- Preserve all 512 context bit-planes through every block.
- Learn only 4-input, 16-bit Boolean truth tables and one discrete source index
  per LUT input.
- Candidate count: 16 fixed candidates per LUT input.
- `mixed` routing is the existing deterministic global candidate baseline.
- `sequence_causal` routing offers recent historical character planes,
  deterministic older positions, and same-class previous vote planes. Every
  source is in the 64-character history; there is no future-token input.
- Hardening: hard-ST training, direct argmax, and per-block hard freeze.
- Readout: fixed interleaved integer GroupSum over 65 character classes.
- Deployment contains no embedding matrix, attention/MLP weights, learned dense
  integer matrix, or real-valued operator. Float training shadows are not
  exported.

## Capacity ladder

Each row uses one LUT layer per block.

| id | routing | votes/character | state bits | blocks | hard gates |
| --- | --- | ---: | ---: | ---: | ---: |
| mixed-v32-d2 | mixed | 32 | 2,592 | 2 | 4,160 |
| causal-v32-d2 | sequence_causal | 32 | 2,592 | 2 | 4,160 |
| causal-v64-d2 | sequence_causal | 64 | 4,672 | 2 | 8,320 |
| causal-v64-d4 | sequence_causal | 64 | 4,672 | 4 | 16,640 |
| causal-v128-d4 | sequence_causal | 128 | 8,832 | 4 | 33,280 |

The matched `mixed-v32-d2` and `causal-v32-d2` runs first use a 20,000-window,
4-epoch/block smoke. Full rows use at most 20 epochs/block, at least 6 epochs,
and validation-hard-accuracy patience 4. The largest row runs only if the
preceding width or depth comparison improves validation hard accuracy.

## Optimization

- AdamW, learning rate 0.03, weight decay 0.
- Cosine decay to 5% of the starting learning rate.
- Batch size 256; evaluation/strict replay batch size 256.
- Truth/wiring temperature: exponential 1.5 to 0.5 within each block.
- GroupSum temperature: square root of votes per character.
- Class-balanced vote-code loss weight 0.25.
- Wiring constraint weight 0.001, fanout cap 8.
- Truth transition cost weight 0.01.
- Gradient clipping: L2 norm 5.

## Metrics and gates

Report soft/hard next-character accuracy, cross-entropy, perplexity and
bits-per-character as analysis metrics; deployment emits only integer logits.
Also report hard top-5 accuracy, train hard accuracy, gap, convergence time,
unused/inactive gates, source routing, structural context support, gate count,
depth, fanout, and payload bits. Integer unigram/bigram/trigram count models are
non-neural reference baselines.

A scale step succeeds only if validation hard accuracy improves. Test-only
gains do not count. Every accepted run must pass strict payload validation,
operator dtype audit with zero real tensors, and exact hardened-carrier logits.
Seed 0 is a screen; no method is promoted without seeds 1 and 2.
