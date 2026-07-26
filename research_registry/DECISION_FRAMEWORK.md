# Discrete learnable-logic decision framework

## Objective

The only mainline objective is useful hard accuracy from a trainable model whose
exported inference contains no real-valued tensor or operator. Training-time
floating shadows and surrogate gradients are allowed; deployment is not.

The primary metric is hard accuracy under a strict Boolean/integer executor.
Soft accuracy and the discretization gap are diagnostics. A method with a tiny
gap and chance-level hard accuracy is a failed method.

## Two independent gaps

1. The discretization gap is the mismatch between a training surrogate and the
   selected hard function. Gumbel-ST, annealing, hard-ST, and block refitting
   address this gap.
2. The hard-capacity gap is the difference between the best useful classifier
   and what the hardened architecture can represent and optimize. State
   cardinality, early bottlenecks, wiring coverage, spatial hierarchy, and
   training budget control this gap.

The accumulated evidence shows that the second gap is now dominant. Width-8192
BitState reaches a 0.05 percentage-point gap at only 27.88% hard CIFAR-10
accuracy. COMBINE hard evaluation can exceed its legacy soft evaluation while
remaining near 36% test accuracy. Neither is discretization collapse.

## Current hierarchy

### Integer accuracy reference, not the logic-native mainline

`full_discrete_a8/d12e192` is the current strict integer accuracy reference. It retains A8 integer state,
uses Wmag7 shift-add projections, exact XNOR-popcount Top-K, integer V
aggregation, integer residual/requantization, and an A16 classifier. Its strict
executor reproduces 76.10% on the complete 5,000-image validation split while
auditing 1,960,214 Torch operations with zero floating or complex tensors.
Twenty audited rows match the QAT carrier logits exactly. The exported payload
contains only uint8, int16, int32, and int64 tensors.

This is a transaction-level integer reference, not cycle-accurate RTL. Dense
shift-add projections are its largest capacity and hardware cost. The result is
one seed: compared with the matched d6/e192 run, d12 doubles parameters, takes
1.706x training wall time, and improves final validation accuracy by 0.80 pp.
It establishes a stronger strict checkpoint, not a statistically robust depth
scaling law. Because learned dense Wmag7 projections dominate its capacity, it
is not eligible as the final LUT/gate/wiring-only method.

### Current logic-native scaling candidate

`bitplane_lut_argmax` preserves all 512 input bit-planes, trains two sequential
banks of hard-forward 4-input LUTs with learned candidate wiring, freezes each
block, and deploys only Boolean truth bits, integer source indices, and fixed
integer GroupSum. On the registered three-seed digits screen, increasing the
learned vote region from 160 to 320 bits improves validation hard accuracy from
88.02% to 91.48% and test hard accuracy from 85.31% to 89.75%. Every paired
validation seed improves, mean accuracy gap falls by 0.62 pp, all learned vote
planes remain active, and the standalone executor observes no real-valued
tensors.

This is the first positive logic-native width result in the integrated tree,
but it is bounded digits evidence, not a CIFAR-scale claim. Training genuinely
changes wiring: 65.27%-73.67% of exported gate inputs leave their default
candidate route. Independent truth-table refitting is rejected because it
reduces validation hard accuracy by 1.67 pp on average. Coordinate-greedy
wiring refit changes no additional source and exactly matches the rejected
truth-refit result.

ABC confirms that every learned 4-input truth table can be lowered to gates,
but the present function population is not strongly table-compressible. Across
the 18 payloads, 76%-79% of tables depend on all four inputs and only 9%-11%
depend on at most two. A unique-table dictionary increases storage by 17%-22%.
Whole-network `strash; dc2` removes roughly 38%-40% of the direct AIG AND
expansion, but comparable K=4 mapping removes only 2.7%-6.9% of LUT nodes and
raises mapped depth from two to three. All exports pass random-vector replay
and ABC CEC. This synthesis scope excludes fixed GroupSum and argmax.

### Pure Boolean count-message screen

`bitstate_count_message_v0` keeps the 192 raw patch threshold bits, retains the
existing local Boolean blocks and stable XNOR Top-K routing, and replaces part
of every one-bit majority message with multiple integer-count threshold bits.
The exported executor reads only Boolean/integer payloads. The matched
three-seed screen is complete and rejects the method: count encoding changes
12.30% of message bits, but mean best-validation hard accuracy falls by
0.05 pp, the test gap grows, and 92.98% of merge lanes remain exact state
identity. This is evidence that the signal is generated but not used, not an
inactive implementation.

### Baselines retained for interpretation

- DLGN argmax: required discretization-gap failure baseline.
- DLGN annealing: tests simple sharpening.
- Gumbel-ST: strongest direct Mind-the-Gap baseline.
- Block Hard-LGN refit: tests frozen-prefix mismatch removal.
- COMBINE Stage598: reproducible hard-patch training baseline, not a strict
  no-real-value full model.
- Attention-clean: real-valued optimization/accuracy ceiling only.

## Components not to repeat

| component | evidence-based disposition | revisit condition |
| --- | --- | --- |
| post-bottleneck width duplication | rejected as information-neutral | only after removing the 18-bit patch neck and with matched capacity controls |
| width-8192 one-bit BitState | gap win, capacity failure | only with richer per-block state and three seeds |
| repeated local shared logic tree | hard tables stayed identity | only with demonstrated hard-table changes and a strict executor |
| fixed Hadamard global mixer | loses matched content Top-K probes | only as a hardware-cost control |
| 72 MiB A8 global LUT tree | sparse coverage and negative 1k result | only with an address-coverage/sample-efficiency argument |
| aggressive Gumbel in BitState | low-gap accuracy collapse | only with a protocol faithful to the original paper or a diagnosed gradient fix |
| anti-literal pressure alone | changes distributions without reliable accuracy | only as a scoped addition to an architecture that already learns useful logic |
| per-block count messages with the existing merge | active signal but no validation gain; merge remains state identity | only with a new hard architecture that guarantees useful message preservation, not another penalty or threshold sweep |
| larger local LUT alone | accuracy can improve but synthesis cost rises sharply | require matched cost-normalized results and ABC equivalence |
| independent per-output empirical truth refit | lowers task-level hard accuracy despite a smaller local mismatch | only with joint class-margin-aware fitting across interacting votes |
| coordinate-greedy post-training wiring refit | changes zero exported sources after training | only with beam/global search or a task-aware objective that can cross coordinate barriers |

## Required capacity report

Every run must report both train-time and deployed capacity:

- trainable shadow parameters;
- hardened payload bits, including tables and wiring;
- learned gate count and selected-function distribution;
- logic depth and the definition used for fixed arithmetic networks;
- maximum fanout;
- input patch information bits and the earliest post-encoder bottleneck;
- state width, bits per state value, storage bits per token, and token count;
- Q/K bit width, Top-K, count/message levels, and tie rate;
- fixed XNOR/popcount/value-count work per sample;
- sampled state entropy, constant ratio, duplicate ratio, and flip rate;
- unused, literal, and nontrivial learned-gate ratios;
- wall time, peak memory, and synthesis statistics when available.

Learned-gate depth must not be presented as total hardware depth when Top-K,
popcount, comparison, or requantization networks are omitted. Such values must
be labeled lower bounds or algorithmic stages.

Dynamic audit operation count is also not a hardware-work metric: one Torch
dispatch may cover an entire batch, so the count changes with batch size. It
proves dtype coverage only. Fixed XNOR/count work, logical coefficients,
bit-level lowering, and synthesis results carry the capacity/cost claim.

## Evidence ladder

1. **Correctness:** unit tests, all 16 gates, deterministic ties, exact hard
   carrier versus payload executor, static no-real source check.
2. **Bounded training smoke:** finite gradients, all gate families receive
   gradient, no immediate state or class collapse.
3. **Registered screen:** at least three paired seeds, validation selection,
   same architecture/data/budget except one declared variable.
4. **Full-data confirmation:** full CIFAR-10, at least three paired seeds,
   complete curves and convergence-time comparison.
5. **Strict deployment replay:** full selected split through the exported
   Boolean/integer executor with a nonzero all-operator audit and zero real
   tensors.
6. **Logic cost:** Boolean lowering, equivalence, gate/depth/fanout, synthesis
   runtime, and PPA assumptions.

A candidate advances one level at a time. A failed screen remains registered as
negative evidence and is not silently retuned on the test set.

## Promotion decision

The policy in `promotion_policy.json` requires three paired seeds, at least two
seed wins, at least 0.20 pp mean hard-accuracy improvement, no seed regression
beyond two percentage points, no material average gap increase, healthy finite
training, validation-only selection, complete capacity reporting, and an
operator-audited Boolean/integer runtime.

The count-message result pauses pure one-bit BitState work. FullDiscrete remains
an accuracy carrier and integer reference, while the logic-native mainline is:

1. retain the 320-vote `bitplane_lut_argmax` result as the registered digits
   width winner and reproduce it from the hashed payloads;
2. replace independent refit with a joint class-margin-aware hard refitter that
   is accepted only if it beats direct argmax on paired validation seeds;
3. continue vote-width scaling with matched payload/gate budgets and stop when
   hard accuracy, utilization, or fanout no longer improves;
4. add fixed spatial routing or shared local LUT structure before CIFAR-small,
   while keeping every persistent state as explicit Boolean bit-planes;
5. confirm the surviving architecture on binarized MNIST and CIFAR-small before
   a full CIFAR-10 budget;
6. export BLIF/Verilog and run ABC/Yosys only after the unsynthesized hard model
   passes accuracy and no-real-value gates.

No new run may reintroduce a learned dense numeric matrix, treat marginal
entropy as task information, or claim success from gap reduction while hard
accuracy regresses.
