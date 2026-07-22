# Hard-LGN v2 design and evidence audit

## Research question

The experiment tests whether fitting and freezing a hard block before training
the next block reduces deployment mismatch under the same wiring, width, depth,
optimizer budget, data split, and random seed as DLGN and Gumbel-ST.

Two gaps must remain separate:

- `acc_gap = |soft_acc - discrete_acc|` is the conventional full relaxed versus
  full hard comparison used for the Mind-the-Gap baseline.
- `path_acc_gap` compares the method-native path with its deployed hard path.
  For block methods this is a hard frozen prefix plus the final relaxed block.

A block method can improve deployable accuracy while worsening the
counterfactual all-relaxed network. Both values are reported; neither is
silently substituted for the other.

## Forward-aligned baselines

`hard_st` uses deterministic argmax in the forward pass and a softmax
derivative in the backward pass. `hard_st_cage` and `gumbel_st_cage` implement
the confidence-adaptive backward temperature from *Align Forward, Adapt
Backward*:

```text
c(t) = mean_n max_i softmax(z_n(t))_i
c_ema = beta * c_ema + (1 - beta) * c(t)
tau_b = tau_max - (tau_max - tau_min) * (c_ema - 1/K) / (1 - 1/K)
```

The defaults match the paper: `tau_max=3.0`, `tau_min=0.5`, `beta=0.99`, and
`K=16`. Temperature affects only the ST derivative for deterministic Hard-ST;
its forward result is identical to deployment for every temperature.

## Task-aware truth-table refit

The original refit independently chose the lowest-MSE Boolean function for
each gate. That cannot exploit error cancellation under GroupSum and does not
optimize the downstream task. The new fitter builds three hard candidates:

1. direct argmax;
2. independent local truth-table refit;
3. coordinate-refined truth tables chosen from each gate's top local
   candidates.

For candidate hard block `H_o`, relaxed block `S`, GroupSum `G`, and a
training-only calibration split, coordinate updates minimize:

```text
J(o) = CE(y, G(H_o(x)))
     + lambda_distill * MSE(G(H_o(x)), G(S(x)))
     + lambda_local * mean_gate_MSE(H_o(x), S(x))
     + lambda_inactive * inactive_gate_ratio(H_o(x))
```

The final argmax/local/task candidate is selected on a deterministic holdout
from the training set. Test labels are never used for refitting or selection.
For small input widths, local reconstruction enumerates the exact input truth
table; otherwise it samples the hard-prefix training distribution.

## Budget and diagnostics

`--block-total-epochs E` distributes exactly `E` epochs across depth, so an
end-to-end `E`-epoch run and a block-wise run process the same number of full
training-set epochs. The historical per-block budget remains available through
`--block-epochs` when `--block-total-epochs` is omitted.

Outputs include the required final table plus:

- `selection_confidence` and backward `tau` per epoch;
- argmax/local/task validation accuracy, loss, teacher-logit MSE, local MSE,
  inactive ratio, chosen candidate, and coordinate updates per block;
- representation MAE/MSE and binary flip ratio after every layer prefix;
- optional BLIF/ABC pre/post gate and depth statistics.

## Evidence available before formal sweeps

The equal-budget quick run is a code-path diagnostic, not a paper result. On
`random_sparse8`, task-first refitting reached `93.75%` hard accuracy versus
`64.06%` for independent refit, but had a larger relaxed-full gap. Balanced
selection restored the `3.12%` full gap of independent refit and the same hard
accuracy. This confirms the intended tradeoff is measurable rather than
establishing a win.

The server-210 Full-K results provide a stronger architectural signal:
multi-level popcount propagation stays within `0.1%` of its binarized-MNIST
teacher at `K=16`, while converting every hidden layer back to one bit loses
`9.6%`. The next model therefore carries a persistent bit-state representation
across depth and treats one-bit blocks as local primitives, not the global
hidden-state format.

## References

- [Mind the Gap: Removing the Discretization Gap in Differentiable Logic Gate Networks](https://arxiv.org/abs/2506.07500)
- [Align Forward, Adapt Backward: Closing the Discretization Gap in Logic Gate Networks](https://arxiv.org/abs/2603.14157)
- [Deep Differentiable Logic Gate Networks](https://arxiv.org/abs/2210.08277)
- [Convolutional Differentiable Logic Gate Networks](https://arxiv.org/abs/2411.04732)
