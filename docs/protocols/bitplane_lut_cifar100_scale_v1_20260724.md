# Bit-plane LUT CIFAR-100 scale protocol v1

Status: frozen before any CIFAR-100 training result is inspected.

## Question

Measure the accuracy ceiling of the current logic-native A8 bit-plane LUT
method when only Boolean LUT count, Boolean depth, and fixed candidate routing
are scaled. No convolution, normalization, real-valued embedding, learned
integer matrix, or test-set model selection is allowed.

## Data

- Dataset: torchvision CIFAR-100, official 50,000 train and 10,000 test rows.
- Split seed: 20260724.
- Training/validation: fixed stratified 45,000/5,000 split of official train.
- Input ABI: raw 32x32x3 uint8, flattened to 3,072 A8 symbols and exactly
  24,576 little-endian Boolean planes. No normalization or learned stem.
- Training augmentation: integer-only reflect-pad/crop by four pixels and
  horizontal flip. Validation/test use unmodified uint8 inputs.
- The official test split is report-only and must not select architecture,
  epoch, routing policy, or hyperparameters.

## Method boundary

- Preserve all 24,576 raw input planes through every block.
- Learn 4-input, 16-bit Boolean truth tables and one discrete source index per
  LUT input.
- Candidate count: 16 fixed candidates per LUT input.
- Hardening: direct argmax and per-block freeze, the registered digits winner.
- Readout: fixed interleaved integer GroupSum over class votes.
- Deployment payload may contain only Boolean tables, integer source indices,
  and fixed Boolean/integer support logic. Training shadow logits are real and
  are never exported.

## Capacity ladder

The screening seed is 0. Each row uses one LUT layer per block.

| id | routing | votes/class | vote LUTs/layer | blocks | hard gates |
| --- | --- | ---: | ---: | ---: | ---: |
| flat-v64-d2 | mixed | 64 | 6,400 | 2 | 12,800 |
| spatial-v64-d2 | image_spatial | 64 | 6,400 | 2 | 12,800 |
| spatial-v128-d2 | image_spatial | 128 | 12,800 | 2 | 25,600 |
| spatial-v128-d4 | image_spatial | 128 | 12,800 | 4 | 51,200 |
| spatial-v256-d4 | image_spatial | 256 | 25,600 | 4 | 102,400 |

Before full runs, `flat-v64-d2` and `spatial-v64-d2` run a 5,000-row,
4-epoch/block smoke. Full runs use at most 30 epochs/block, at least 8 epochs,
and validation-hard-accuracy patience 5.

## Optimization

- AdamW, learning rate 0.03, weight decay 0.
- Cosine decay to 5% of the starting learning rate.
- Batch size 128; evaluation/strict replay batch size 128.
- Truth/wiring temperature: exponential 1.5 to 0.5 within each block.
- GroupSum temperature: square root of votes per class.
- Code loss weight 0.25.
- Wiring constraint weight 0.001, fanout cap 8.
- Truth transition cost weight 0.01.
- Gradient clipping: L2 norm 5.

## Gates

Smoke is valid only if gradients are finite, payload validation passes, strict
Boolean/integer logits exactly match the hardened carrier, runtime float tensor
count is zero, and validation hard accuracy exceeds the 1% chance level.

The capacity ladder is interpreted using validation hard accuracy, soft/hard
gap, convergence time, unused gates, depth, fanout, and logical payload. A
larger model is a scaling success only if validation hard accuracy improves;
test-only gains do not count. The best validation configuration is repeated on
seeds 1 and 2 before promotion.
