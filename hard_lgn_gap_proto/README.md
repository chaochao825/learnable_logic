# Block-wise Hard-LGN Gap Prototype

PyTorch prototype for comparing standard differentiable logic gate networks
against block-wise hard refitting.

The script intentionally uses the same 16 two-input Boolean gate ordering as
`difflogic.functional` from `/home/spco/convlogic`, fixed random wiring, and a
GroupSum output head. It adds the missing experiment paths needed for the
Hard-LGN question:

- standard relaxed DLGN with final argmax discretization
- DLGN with temperature and entropy annealing
- Gumbel-Softmax / straight-through LGN
- block-wise relaxed training with final argmax discretization
- block-wise Hard-LGN with per-block truth-table refitting

Run on the 210 server:

```bash
cd /home/spco/sow_linear/hard_lgn_gap_proto
source /home/wangmeiqi/anaconda3/etc/profile.d/conda.sh
conda activate convlogic
PYTHONPATH=/home/spco/convlogic/src:$PYTHONPATH CUDA_VISIBLE_DEVICES=0 \
  python hard_lgn_benchmark.py --quick --out-dir runs/quick
```

Check compatibility with the installed `/home/spco/convlogic` difflogic
primitives before running comparisons:

```bash
PYTHONPATH=/home/spco/convlogic/src:$PYTHONPATH \
  python hard_lgn_benchmark.py \
  --difflogic-compat-check --compat-only \
  --out-dir runs/difflogic_compat_v1
```

This writes `difflogic_compat.json` and verifies the 16-gate ordering,
soft weighted gates, GroupSum, random wiring, and hard/soft `LogicLayer`
forward behavior against the installed difflogic implementation. The benchmark
keeps local layer classes so block-wise training and truth-table refitting can
inspect and replace individual blocks directly.

LightLogic-first baseline and Goal 1/2 sweep:

```bash
python lightlogic_experiments.py \
  --datasets parity8 majority9 random_sparse10 \
  --methods light_iwp light_iwp_anneal light_iwp_st light_iwp_gumbel_st dlgn_op \
  --seeds 0 \
  --width 128 \
  --layers 4 \
  --epochs 120 \
  --batch-size 256 \
  --eval-batch-size 2048 \
  --lr 0.01 \
  --estimator sigmoid \
  --init residual \
  --temp-start 2.0 \
  --temp-end 0.1 \
  --entropy-weight 0.001 \
  --out-dir runs/lightlogic_bool_seed0_goal012_sigmoid_v1
```

Sinusoidal-estimator LightLogic sweep:

```bash
python lightlogic_experiments.py \
  --datasets parity8 majority9 random_sparse10 \
  --methods light_iwp light_iwp_st light_iwp_gumbel_st dlgn_op \
  --seeds 0 \
  --width 128 \
  --layers 4 \
  --epochs 120 \
  --batch-size 256 \
  --eval-batch-size 2048 \
  --lr 0.01 \
  --estimator sinusoidal \
  --init residual \
  --out-dir runs/lightlogic_bool_seed0_goal012_sinusoidal_v1
```

Sklearn digits LightLogic baseline:

```bash
python lightlogic_experiments.py \
  --datasets digits \
  --methods light_iwp light_iwp_st dlgn_op \
  --seeds 0 \
  --width 320 \
  --layers 4 \
  --epochs 150 \
  --batch-size 256 \
  --eval-batch-size 2048 \
  --lr 0.01 \
  --estimator sinusoidal \
  --init residual \
  --out-dir runs/lightlogic_digits_seed0_goal02_sinusoidal_v1
```

Merge current LightLogic evidence:

```bash
python build_lightlogic_report.py \
  --runs-root runs \
  --runs lightlogic_bool_seed0_goal012_sigmoid_v1 lightlogic_bool_seed0_goal012_sinusoidal_v1 lightlogic_digits_seed0_goal02_sinusoidal_v1 \
  --out-dir runs/lightlogic_report_goal012_v5
```

Current LightLogic evidence is deliberately conservative: Goal 0 is covered by
plain `light_iwp` rows with continuous/discrete accuracy, discretization gap,
gate utilization, gate count, parameter count, and training time. Goal 1 and
Goal 2 are comparison rows only; current annealing evidence is mixed and must
be checked for gate-utilization collapse.

K-gate truth-table expansion and calibration:

```bash
python lightlogic_k_expansion.py \
  --datasets parity8 majority9 random_sparse10 \
  --seeds 0 \
  --k-values 2 4 8 16 32 \
  --width 128 \
  --layers 4 \
  --epochs 120 \
  --batch-size 256 \
  --eval-batch-size 2048 \
  --lr 0.01 \
  --estimator sinusoidal \
  --init residual \
  --out-dir runs/lightlogic_k_expansion_bool_seed0_v1

python lightlogic_k_expansion.py \
  --datasets parity8 majority9 random_sparse10 \
  --seeds 0 \
  --k-values 2 4 8 16 32 \
  --width 128 \
  --layers 4 \
  --epochs 120 \
  --batch-size 256 \
  --eval-batch-size 2048 \
  --lr 0.01 \
  --estimator sinusoidal \
  --init residual \
  --calibration-mode per_layer \
  --out-dir runs/lightlogic_k_expansion_bool_seed0_per_layer_v1

python lightlogic_k_expansion.py \
  --datasets parity8 majority9 random_sparse10 \
  --seeds 0 \
  --k-values 2 4 8 16 32 \
  --width 128 \
  --layers 4 \
  --epochs 120 \
  --batch-size 256 \
  --eval-batch-size 2048 \
  --lr 0.01 \
  --estimator sinusoidal \
  --init residual \
  --calibration-mode per_neuron \
  --out-dir runs/lightlogic_k_expansion_bool_seed0_per_neuron_v1
```

Merge K-expansion evidence:

```bash
python build_lightlogic_k_report.py \
  --runs-root runs \
  --runs lightlogic_k_expansion_bool_seed0_v1 lightlogic_k_expansion_bool_seed0_per_layer_v1 lightlogic_k_expansion_bool_seed0_per_neuron_v1 \
  --out-dir runs/lightlogic_k_report_goal34_v4
```

Current K-expansion evidence: Goal 3 is implemented and swept for K=2,4,8,16,32.
Local q-vs-r/K error decreases with larger K, but full-network accuracy does
not necessarily improve after thresholding back to one bit per layer. Goal 4
calibration is mixed: current per-layer/per-neuron calibration wins 4/30
same-K comparisons.

Fuller Boolean sweep:

```bash
PYTHONPATH=/home/spco/convlogic/src:$PYTHONPATH CUDA_VISIBLE_DEVICES=0 \
  python hard_lgn_benchmark.py \
  --datasets parity8 majority9 random_sparse10 \
  --methods dlgn dlgn_anneal gumbel_st block_relaxed block_hard_refit \
  --epochs 80 --block-epochs 50 --width 64 --layers 4 --seeds 0 1 2 \
  --out-dir runs/bool_sweep
```

Binarized MNIST smoke using flattened thresholded pixels:

```bash
PYTHONPATH=/home/spco/convlogic/src:$PYTHONPATH CUDA_VISIBLE_DEVICES=0 \
  python hard_lgn_benchmark.py \
  --datasets mnist --data-dir /home/spco/data --download-data \
  --methods dlgn dlgn_anneal gumbel_st block_relaxed block_hard_refit \
  --epochs 20 --block-epochs 12 --width 400 --layers 3 --seeds 0 \
  --image-max-train 4000 --image-max-test 1000 \
  --out-dir runs/mnist_smoke
```

Small CIFAR-10 smoke using the existing `/home/spco/data/cifar-10-batches-py`:

```bash
PYTHONPATH=/home/spco/convlogic/src:$PYTHONPATH CUDA_VISIBLE_DEVICES=0 \
  python hard_lgn_benchmark.py \
  --datasets cifar10_small --data-dir /home/spco/data \
  --methods dlgn dlgn_anneal gumbel_st block_relaxed block_hard_refit \
  --epochs 10 --block-epochs 6 --width 1600 --layers 3 --seeds 0 \
  --image-max-train 2000 --image-max-test 500 \
  --out-dir runs/cifar10_smoke
```

Optional ABC structural stats for small hard networks:

```bash
PYTHONPATH=/home/spco/convlogic/src:$PYTHONPATH CUDA_VISIBLE_DEVICES=0 \
  python hard_lgn_benchmark.py --quick --abc-stats \
  --abc-path /home/spco/boolean_sat/abc/abc \
  --out-dir runs/quick_abc
```

Outputs:

- `results.csv`: required method/dataset metric table
- `summary.md`: Markdown version of the comparison table plus run metadata
- `per_epoch.csv`: convergence traces for soft and discrete metrics
- `block_diagnostics.csv`: per-block path-soft versus frozen-hard metrics plus
  refit MSE versus argmax MSE
- `layer_diagnostics.csv`: per-layer relaxed-path versus hard-path
  representation mismatch metrics for depth-wise gap accumulation checks
- `synthesis_stats.csv`: optional BLIF/ABC structural statistics when
  `--abc-stats` is enabled
- `soft_inference_samples_per_sec` / `discrete_inference_samples_per_sec`:
  optional PyTorch forward-pass throughput columns in `results.csv` when
  `--inference-bench` is enabled. These are not bit-packed Boolean inference
  kernels and should not be compared directly to the DLGN paper's optimized
  CPU inference numbers.
- `aggregate_results.py`: optional post-processing for seed means, standard
  deviations, DLGN/Gumbel deltas, and refit diagnostics
- `sweep_gumbel.py`: optional runner for reproducible Gumbel-ST hyperparameter
  sweeps and best-config summaries. It defaults to `gumbel_st`; `--method
  gumbel_soft` is available only as a diagnostic Gumbel-Softmax relaxation
  without straight-through hard forward.
- `build_tuned_comparison.py`: rebuilds the seed-0 tuned Gumbel comparison
  table from the ABC baseline run and Gumbel sweep summaries.
- `build_required_table.py`: consolidates the latest aggregate outputs into
  the exact user-requested metrics table and a provenance companion.
- `build_gap_taxonomy.py`: separates relaxed-full gap, method-native path gap,
  and per-block refit error so block-wise claims are not conflated with the
  traditional Mind-the-Gap metric.
- `next_stage_protocol.md`: experiment protocol for the next deployable
  hard-path-accuracy stage: random wiring search, Boolean registers, and
  redundancy-aware hardening.
- `lightlogic_experiments.py`: LightLogic-first runner for input-wise
  parametrization (IWP), OP/DLGN comparison, annealing/entropy, and ST/Gumbel
  training under the same fixed wiring and metric schema.
- `build_lightlogic_report.py`: merges LightLogic runs into goal-level evidence
  tables and reports whether Goal 0/1/2 are supported by the current artifacts.
- `lightlogic_k_expansion.py`: Goal 3/4 prototype for per-gate K-hard-gate
  truth-table expansion plus fixed, per-layer, or per-neuron threshold
  calibration.
- `build_lightlogic_k_report.py`: merges K-expansion and calibration runs into
  best-row, calibration-delta, and goal-check tables.
- `wiring_search.py`: Part A prototype for random connectivity search followed
  by fixed-wiring retraining. It separates wiring seed from gate-logit init
  seed and selects wiring on validation hard accuracy before held-out test
  reporting.
- `sequential_register_lgn.py`: Part B prototype for synchronous Boolean
  register LGNs on controlled sequence tasks, with feedforward LGN and small
  RNN/GRU baselines.
- `redundant_hardening.py`: Part C prototype for redundant vote bits, margin /
  activity / diversity / dropout regularization, hard-accuracy checkpointing,
  and validation-selected hardening candidates.
- `build_next_stage_tables.py`: current-evidence report builder for the pasted
  next-stage goal. It combines available Part A/B/C runs into the required
  leaderboard, comparison tables, success checks, and coverage audit without
  claiming that smoke-only evidence completes the full goal.
- `summarize_gap_report.py`: builds a consolidated Mind-the-Gap-style report
  from timing-aware aggregate outputs and tuned Gumbel comparison rows. It
  keeps multi-seed fixed-Gumbel aggregates separate from seed-0 tuned-Gumbel
  checks.
- `summarize_synthesis.py`: joins optional `synthesis_stats.csv` rows with
  `results.csv` and reports ABC structural changes. It does not re-import
  optimized ABC nets for post-synthesis accuracy.
- `evaluate_abc_blif.py`: writes optimized BLIF files with ABC and evaluates
  source/optimized BLIF networks on Boolean test sets using the prototype's
  GroupSum classification rule. This provides post-ABC hard accuracy/gap
  checks plus BLIF graph fanout and unused-node metrics for small Boolean
  runs.

Aggregate outputs:

- `aggregate_by_method_dataset.csv`: seed mean/std for every method and dataset.
- `comparison_vs_baselines.csv`: absolute deltas versus DLGN/Gumbel; percentage
  gap reductions are reported as `nan` when the DLGN gap is below `1e-2`.
- `block_diagnostics_all_blocks_aggregate.csv`: average over every block stage.
- `block_diagnostics_by_block.csv`: per-block averages across seeds.
- `block_diagnostics_final_block.csv`: final-block path diagnostics comparable
  with the final `path_*` metrics in `results.csv`.
- `layer_diagnostics_all_layers_aggregate.csv`: average representation mismatch
  over all layer prefixes.
- `layer_diagnostics_by_layer.csv`: depth-wise soft/hard representation
  mismatch by layer prefix.
- `layer_diagnostics_final_layer.csv`: final-layer representation mismatch,
  closest to the final output gap measurement.

Aggregate rebuild including layer diagnostics:

```bash
python aggregate_results.py \
  --results runs/bool_seed0_layerdiag_v1/results.csv \
  --block-diagnostics runs/bool_seed0_layerdiag_v1/block_diagnostics.csv \
  --layer-diagnostics runs/bool_seed0_layerdiag_v1/layer_diagnostics.csv \
  --out-dir runs/bool_seed0_layerdiag_v1/aggregate
```

Optional PyTorch inference-throughput smoke:

```bash
PYTHONPATH=/home/spco/convlogic/src:$PYTHONPATH CUDA_VISIBLE_DEVICES=0 \
  python hard_lgn_benchmark.py \
  --datasets parity8 majority9 \
  --methods dlgn dlgn_anneal gumbel_st block_relaxed block_hard_refit \
  --epochs 20 --block-epochs 12 --width 64 --layers 4 --seeds 0 \
  --inference-bench --inference-bench-device cpu --inference-bench-repeats 20 \
  --out-dir runs/bool_seed0_inferbench_v1
```

Generic tuned Gumbel comparison rebuild:

```bash
python build_tuned_comparison.py \
  --runs-root runs \
  --out-dir runs/gumbel_tuned_comparison_seed0
```

Timing-v3 seed-0 comparison rebuild:

```bash
python build_tuned_comparison.py \
  --runs-root runs \
  --out-dir runs/gumbel_tuned_comparison_seed0_timing_v3 \
  --base-run bool_seeds012_timing_v3 \
  --gumbel-st-run gumbel_sweep_bool_seed0_timing_v3 \
  --seeds 0 \
  --skip-gumbel-soft
```

The comparison table keeps both `method` and `source_method`: `method` is the
comparison label, while `source_method`, `source_run`, and `source_file` identify
the exact source row. Gumbel sweep rows are selected by highest discrete
accuracy, then lowest accuracy gap, then lowest discrete loss, then lowest
training time, and record that rule in
`selection_criterion`. Use `--seeds 0` when the baseline source run contains
multiple seeds and a seed-0 paper-style comparison is desired. Use
`--skip-gumbel-soft` to omit the diagnostic non-ST Gumbel rows.

For convergence-speed sweeps, `--target-acc-override <value>` can set a common
declared threshold across methods; the value must be in `[0.0, 1.0]`. This is
useful for prototype-scale runs where the built-in dataset target may be
intentionally too high for short budgets.

Required table rebuild for the remote 210 project layout:

```bash
python build_required_table.py \
  --runs-root runs \
  --out-dir runs/reports_required_table_v1
```

This writes `required_metrics_table.csv` with exactly
`method | dataset | soft_acc | discrete_acc | acc_gap | soft_loss |
discrete_loss | loss_gap | train_time | epochs_to_target |
unused_gate_ratio | gate_count | depth | fanout_max`, plus a Markdown copy and
`required_metrics_table_with_provenance.csv`. `epochs_to_target` is averaged
over target-reaching seeds and is `-1` when no seed reaches the target.

Gap taxonomy rebuild for separating the three evaluation views:

```bash
python build_gap_taxonomy.py \
  --runs-root runs \
  --out-dir runs/reports_gap_taxonomy_v1
```

This writes `gap_taxonomy_block_hard_summary.csv`,
`gap_taxonomy_all_methods.csv`, `gap_taxonomy_block_refit_by_block.csv`, and
`gap_taxonomy_report.md`. Use `full_*` columns for the traditional
Mind-the-Gap relaxed-full vs hard-full comparison, `native_*` columns for the
block-wise hard-prefix-trained path, and the block refit table to locate
per-block fitting error.

Random wiring search smoke:

```bash
python wiring_search.py \
  --datasets parity6 majority7 \
  --base-method dlgn \
  --num-candidates 4 \
  --search-epochs 4 \
  --final-epochs 8 \
  --width 32 \
  --layers 3 \
  --seeds 0 \
  --matched-budget-baseline \
  --out-dir runs/wiring_search_smoke_v1
```

This writes `wiring_search_candidates.csv`,
`wiring_search_results.csv`, and `wiring_search_report.md`. Use
`fixed_random_matched_budget` when claiming a search benefit under comparable
training budget; otherwise `fixed_random` is only a no-search reference.

Boolean random wiring search sweep:

```bash
python wiring_search.py \
  --datasets parity8 majority9 random_sparse10 \
  --base-method dlgn \
  --num-candidates 8 \
  --seeds 0 \
  --width 64 \
  --layers 4 \
  --search-epochs 10 \
  --final-epochs 40 \
  --matched-budget-baseline \
  --batch-size 256 \
  --eval-batch-size 2048 \
  --out-dir runs/wiring_search_bool_sweep_v1
```

This is the seed-0 Part A Boolean-task wiring sweep. It covers parity,
majority, and random sparse Boolean tasks with a matched-budget fixed-random
baseline. In this run, random wiring search does not beat the matched-budget
fixed-random baseline.

Boolean candidate-set learnable connectivity check:

```bash
python wiring_search.py \
  --datasets parity8 majority9 random_sparse10 \
  --base-method dlgn \
  --num-candidates 8 \
  --candidate-set-learnable \
  --candidate-set-size 4 \
  --seeds 0 \
  --width 64 \
  --layers 4 \
  --search-epochs 10 \
  --final-epochs 40 \
  --matched-budget-baseline \
  --batch-size 256 \
  --eval-batch-size 2048 \
  --out-dir runs/wiring_candidate_learnable_bool_seed0_v1
```

This run adds `candidate_set_learnable_freeze`: for each gate, it learns a
selector over a fixed candidate set of input pairs, freezes the selected
ordinary wiring, then retrains gate logits on the full training split. It is a
controlled candidate-connectivity baseline, not a broad learnable-connectivity
win claim.

Additional Boolean wiring-search seeds:

```bash
python wiring_search.py \
  --datasets parity8 majority9 random_sparse10 \
  --base-method dlgn \
  --num-candidates 8 \
  --seeds 1 2 \
  --width 64 \
  --layers 4 \
  --search-epochs 10 \
  --final-epochs 40 \
  --matched-budget-baseline \
  --batch-size 256 \
  --eval-batch-size 2048 \
  --out-dir runs/wiring_search_bool_seeds12_v1
```

Together with `wiring_search_bool_sweep_v1`, this gives Boolean seed0/1/2
coverage for the wiring-search comparison. Digits and MNIST remain seed-0
path-coverage runs.

Sklearn digits wiring search:

```bash
python wiring_search.py \
  --datasets digits \
  --base-method dlgn \
  --num-candidates 8 \
  --seeds 0 \
  --width 80 \
  --layers 4 \
  --search-epochs 10 \
  --final-epochs 40 \
  --matched-budget-baseline \
  --batch-size 256 \
  --eval-batch-size 2048 \
  --out-dir runs/wiring_search_digits_v2
```

The width is 80 because the wiring-search prototype requires width to be
divisible by the number of output classes; sklearn digits has 10 classes.

Binarized MNIST wiring-search smoke:

```bash
python wiring_search.py \
  --datasets mnist \
  --data-dir /home/spco/data \
  --image-max-train 1000 \
  --image-max-test 300 \
  --base-method dlgn \
  --num-candidates 4 \
  --seeds 0 \
  --width 400 \
  --layers 3 \
  --search-epochs 5 \
  --final-epochs 15 \
  --matched-budget-baseline \
  --batch-size 256 \
  --eval-batch-size 2048 \
  --out-dir runs/wiring_search_mnist_smoke_v1
```

This is a small MNIST path-coverage run, not a full MNIST-scale wiring-search
claim. The width is 400 to cover the 784-bit thresholded input and remain
divisible by 10 classes.

Sequential register-LGN smoke:

```bash
python sequential_register_lgn.py \
  --quick \
  --out-dir runs/sequential_register_smoke_v1
```

This writes `sequential_results.csv` and `sequential_report.md`. The
`feedforward_lgn` baseline sees only the current input bit at each time step;
`register_lgn_*` methods concatenate current input bits with explicit Boolean
register bits and update registers synchronously. RNN/GRU rows are continuous
state references, so their `gate_count` is not a logic-hardware metric.

Required-task sequential comparison:

```bash
python sequential_register_lgn.py \
  --tasks sequence_parity delayed_copy temporal_majority fsm_endswith101 \
  --methods feedforward_lgn register_lgn_soft register_lgn_st rnn_small gru_small \
  --seeds 0 \
  --n-train 512 \
  --n-test 256 \
  --time-steps 8 \
  --delay 2 \
  --state-bits 8 \
  --width 32 \
  --depth 3 \
  --votes 16 \
  --epochs 40 \
  --batch-size 128 \
  --out-dir runs/sequential_register_required_tasks_v1
```

This covers the four required synthetic sequence tasks and includes both RNN
and GRU baselines under the same seed-0 budget. In the current run,
`register_lgn_soft` solves delayed copy and FSM recognition, improves temporal
majority over feedforward LGN, and does not solve sequence parity.

Multi-seed required-task sequential comparison:

```bash
python sequential_register_lgn.py \
  --tasks sequence_parity delayed_copy temporal_majority fsm_endswith101 \
  --methods feedforward_lgn register_lgn_soft register_lgn_st rnn_small gru_small \
  --seeds 0 1 2 \
  --n-train 512 \
  --n-test 256 \
  --time-steps 8 \
  --delay 2 \
  --state-bits 8 \
  --width 32 \
  --depth 3 \
  --votes 16 \
  --epochs 40 \
  --batch-size 128 \
  --out-dir runs/sequential_register_required_tasks_v2
```

This v2 run repeats the four required sequence tasks over seeds 0, 1, and 2.
It strengthens the delayed-copy and FSM evidence, while showing temporal
majority and sequence parity are not robust solved-task claims for the current
register LGN budget.

Redundant hardening smoke:

```bash
python redundant_hardening.py \
  --quick \
  --out-dir runs/redundant_hardening_smoke_v3
```

This writes `redundant_hardening_results.csv`,
`hardening_candidates.csv`, `checkpoint_trace.csv`, and
`redundant_hardening_report.md`. Candidate hardening is selected by validation
hard accuracy first within each method's eligible candidate subset, then hard
loss / refit / circuit-cost score. The candidate table reports both
`eligible_for_method` and `selection_policy`; only `redundant_task_hardened`
selects over all listed hardening candidates. For this feedforward redundancy
prototype, `native_gap` is reported equal to `full_gap` because there is no
separate hard-prefix path; final deployable `hard_acc` is the primary metric.

Boolean redundancy scaling sweep:

```bash
python redundant_hardening.py \
  --datasets parity8 majority9 random_sparse10 \
  --methods no_redundancy more_gates_only redundant_regularized redundant_task_hardened \
  --redundancy-factors 1 2 4 8 \
  --seeds 0 \
  --base-width 32 \
  --base-votes 8 \
  --layers 3 \
  --epochs 30 \
  --batch-size 256 \
  --eval-batch-size 2048 \
  --best-of-n 8 \
  --out-dir runs/redundant_hardening_bool_sweep_v1
```

This is the seed-0 Part C Boolean-task sweep. It covers parity, majority, and
random sparse Boolean tasks plus the requested 1x/2x/4x/8x redundancy factors
under seed 0. It still does not cover digits/MNIST, best-of-32, or ABC.

Additional Boolean redundancy-hardening seeds:

```bash
python redundant_hardening.py \
  --datasets parity8 majority9 random_sparse10 \
  --methods no_redundancy more_gates_only redundant_regularized redundant_task_hardened \
  --redundancy-factors 1 2 4 8 \
  --seeds 1 2 \
  --base-width 32 \
  --base-votes 8 \
  --layers 3 \
  --epochs 30 \
  --batch-size 256 \
  --eval-batch-size 2048 \
  --best-of-n 8 \
  --out-dir runs/redundant_hardening_bool_seeds12_v1
```

Together with `redundant_hardening_bool_sweep_v1`, this gives Boolean
seed0/1/2 coverage for redundancy-hardening. Digits and MNIST remain seed-0
path-coverage runs.

Truth-table-refit-only final-test sweep:

```bash
python redundant_hardening.py \
  --datasets parity8 majority9 random_sparse10 \
  --methods redundant_truth_refit_only \
  --redundancy-factors 2 4 8 \
  --seeds 0 1 2 \
  --base-width 32 \
  --base-votes 8 \
  --layers 3 \
  --epochs 30 \
  --batch-size 256 \
  --eval-batch-size 2048 \
  --best-of-n 8 \
  --out-dir runs/redundant_truth_refit_bool_seeds012_v1
```

This method uses the same regularized redundant training path as
`redundant_task_hardened`, but its selection policy only allows
`truth_table_refit`. It gives final held-out test rows for truth-table refit,
whereas `truth_table_refit_candidate_comparison.csv` is validation-only.

Sklearn digits truth-table-refit-only final-test sweep:

```bash
python redundant_hardening.py \
  --datasets digits \
  --methods redundant_truth_refit_only \
  --redundancy-factors 2 4 8 \
  --seeds 0 \
  --base-width 40 \
  --base-votes 4 \
  --layers 3 \
  --epochs 30 \
  --batch-size 256 \
  --eval-batch-size 2048 \
  --best-of-n 8 \
  --out-dir runs/redundant_truth_refit_digits_v1
```

Sklearn digits redundancy sweep:

```bash
python redundant_hardening.py \
  --datasets digits \
  --methods no_redundancy more_gates_only redundant_regularized redundant_task_hardened \
  --redundancy-factors 1 2 4 8 \
  --seeds 0 \
  --base-width 40 \
  --base-votes 4 \
  --layers 3 \
  --epochs 30 \
  --batch-size 256 \
  --eval-batch-size 2048 \
  --best-of-n 8 \
  --out-dir runs/redundant_hardening_digits_v1
```

This adds sklearn digits coverage for Part C. It does not replace the Boolean
sweep; the current evidence report merges both.

Binarized MNIST redundancy smoke:

```bash
python redundant_hardening.py \
  --datasets mnist \
  --data-dir /home/spco/data \
  --image-max-train 1000 \
  --image-max-test 300 \
  --methods no_redundancy more_gates_only redundant_regularized redundant_task_hardened \
  --redundancy-factors 1 2 4 \
  --seeds 0 \
  --base-width 400 \
  --base-votes 20 \
  --layers 3 \
  --epochs 10 \
  --batch-size 256 \
  --eval-batch-size 2048 \
  --best-of-n 4 \
  --out-dir runs/redundant_hardening_mnist_smoke_v1
```

This is a small MNIST path-coverage run. It does not cover 8x MNIST redundancy
or best-of-8/32 for MNIST, but the combined report keeps the broader Boolean
and digits sweeps in the same evidence table.

Binarized MNIST truth-table-refit-only final-test smoke:

```bash
python redundant_hardening.py \
  --datasets mnist \
  --data-dir /home/spco/data \
  --image-max-train 1000 \
  --image-max-test 300 \
  --methods redundant_truth_refit_only \
  --redundancy-factors 2 4 \
  --seeds 0 \
  --base-width 400 \
  --base-votes 20 \
  --layers 3 \
  --epochs 10 \
  --batch-size 256 \
  --eval-batch-size 2048 \
  --best-of-n 4 \
  --out-dir runs/redundant_truth_refit_mnist_smoke_v1
```

This mirrors the MNIST smoke budget and remains path-coverage evidence, not a
full MNIST-scale truth-refit claim.

CIFAR-10 small redundancy/truth-refit smoke:

```bash
python redundant_hardening.py \
  --datasets cifar10_small \
  --data-dir /home/spco/data \
  --image-max-train 1000 \
  --image-max-test 300 \
  --methods more_gates_only redundant_regularized redundant_task_hardened redundant_truth_refit_only \
  --redundancy-factors 2 \
  --seeds 0 \
  --base-width 800 \
  --base-votes 40 \
  --layers 3 \
  --epochs 8 \
  --batch-size 256 \
  --eval-batch-size 1024 \
  --best-of-n 4 \
  --out-dir runs/redundant_hardening_cifar10_smoke_v1
```

This is only CIFAR path coverage under a small budget. It is not comparable to
the Mind-the-Gap full CIFAR setting or the 61M-gate convolutional LGN result.

Prior smoke-only next-stage evidence tables:

```bash
python build_next_stage_tables.py \
  --runs-root runs \
  --wiring-run wiring_search_smoke_v1 \
  --sequential-run sequential_register_required_tasks_v1 \
  --redundancy-run redundant_hardening_smoke_v3 \
  --out-dir runs/reports_next_stage_v1
```

This writes `unified_hard_accuracy_leaderboard.csv`,
`wiring_search_comparison.csv`, `register_sequential_comparison.csv`,
`redundancy_scaling_curve.csv`, `hardening_method_comparison.csv`,
`soft_hard_gap_taxonomy_current.csv`, `success_checks.csv`,
`coverage_audit.csv`, and `next_stage_evidence_report.md`. The report is an
audit of the earlier smoke evidence, not a completion claim; it marks missing
datasets, missing redundancy factors, and unsupported success criteria
explicitly.

Prior next-stage evidence tables with the Boolean Part C sweep:

```bash
python build_next_stage_tables.py \
  --runs-root runs \
  --wiring-run wiring_search_smoke_v1 \
  --sequential-run sequential_register_required_tasks_v1 \
  --redundancy-run redundant_hardening_bool_sweep_v1 \
  --out-dir runs/reports_next_stage_v2
```

This report updates the Part C coverage audit to include random sparse Boolean
tasks, 1x/2x/4x/8x redundancy, and best-of-8 Gumbel hardening candidates, but
it still uses the older Part A wiring-search smoke run.

Prior next-stage evidence tables with Boolean Part A/C sweeps:

```bash
python build_next_stage_tables.py \
  --runs-root runs \
  --wiring-run wiring_search_bool_sweep_v1 \
  --sequential-run sequential_register_required_tasks_v1 \
  --redundancy-run redundant_hardening_bool_sweep_v1 \
  --out-dir runs/reports_next_stage_v3
```

This report merges the Boolean wiring and redundancy sweeps, but predates the
sklearn digits runs.

Prior next-stage evidence tables with Boolean + digits A/C runs:

```bash
python build_next_stage_tables.py \
  --runs-root runs \
  --wiring-runs wiring_search_bool_sweep_v1 wiring_search_digits_v2 \
  --sequential-run sequential_register_required_tasks_v1 \
  --redundancy-runs redundant_hardening_bool_sweep_v1 redundant_hardening_digits_v1 \
  --out-dir runs/reports_next_stage_v4
```

This report merges Boolean and sklearn-digits Part A/C evidence, but predates
the small binarized MNIST coverage runs.

Prior next-stage evidence tables with Boolean + digits + MNIST A/C runs:

```bash
python build_next_stage_tables.py \
  --runs-root runs \
  --wiring-runs wiring_search_bool_sweep_v1 wiring_search_digits_v2 wiring_search_mnist_smoke_v1 \
  --sequential-run sequential_register_required_tasks_v1 \
  --redundancy-runs redundant_hardening_bool_sweep_v1 redundant_hardening_digits_v1 redundant_hardening_mnist_smoke_v1 \
  --out-dir runs/reports_next_stage_v5
```

This report merges Boolean, sklearn-digits, and small binarized-MNIST Part A/C
evidence, but predates the Boolean seed1/2 extension.

Prior next-stage evidence tables with Boolean seed0/1/2 plus digits + MNIST
A/C runs:

```bash
python build_next_stage_tables.py \
  --runs-root runs \
  --wiring-runs wiring_search_bool_sweep_v1 wiring_search_bool_seeds12_v1 wiring_search_digits_v2 wiring_search_mnist_smoke_v1 \
  --sequential-run sequential_register_required_tasks_v1 \
  --redundancy-runs redundant_hardening_bool_sweep_v1 redundant_hardening_bool_seeds12_v1 redundant_hardening_digits_v1 redundant_hardening_mnist_smoke_v1 \
  --out-dir runs/reports_next_stage_v6
```

This report merges Boolean seed0/1/2, sklearn-digits seed0, and small
binarized-MNIST seed0 Part A/C evidence. The success checks compare
redundancy-hardening per dataset/seed pair, not by cross-seed best rows.

Prior next-stage evidence tables with strongest redundant-baseline checks:

```bash
python build_next_stage_tables.py \
  --runs-root runs \
  --wiring-runs wiring_search_bool_sweep_v1 wiring_search_bool_seeds12_v1 wiring_search_digits_v2 wiring_search_mnist_smoke_v1 \
  --sequential-run sequential_register_required_tasks_v1 \
  --redundancy-runs redundant_hardening_bool_sweep_v1 redundant_hardening_bool_seeds12_v1 redundant_hardening_digits_v1 redundant_hardening_mnist_smoke_v1 \
  --out-dir runs/reports_next_stage_v7
```

This report adds `redundancy_strong_baseline_comparison.csv`, which compares the best
`redundant_task_hardened` row against the strongest non-task-aware redundant
baseline (`more_gates_only` or `redundant_regularized`) for each dataset/seed
pair. This prevents treating a gain from extra gates alone as evidence for
the task-aware redundant hardening/selection method. It is not a
truth-table-refit-only claim because the selected `redundant_task_hardened`
row may use argmax, Gumbel, or truth-table refit depending on validation
selection.

Prior next-stage evidence tables with truth-table-refit-only candidate
checks:

```bash
python build_next_stage_tables.py \
  --runs-root runs \
  --wiring-runs wiring_search_bool_sweep_v1 wiring_search_bool_seeds12_v1 wiring_search_digits_v2 wiring_search_mnist_smoke_v1 \
  --sequential-run sequential_register_required_tasks_v1 \
  --redundancy-runs redundant_hardening_bool_sweep_v1 redundant_hardening_bool_seeds12_v1 redundant_hardening_digits_v1 redundant_hardening_mnist_smoke_v1 \
  --out-dir runs/reports_next_stage_v8
```

This report adds `truth_table_refit_candidate_comparison.csv`, which compares
`truth_table_refit` against the best non-refit candidate inside
`redundant_task_hardened` for each dataset/seed/factor. This is a validation
candidate comparison, not final test accuracy; final test rows still come from
the selected hardening candidate in `redundancy_scaling_curve.csv`.

Prior next-stage evidence tables with Boolean truth-table-refit-only final-test
rows:

```bash
python build_next_stage_tables.py \
  --runs-root runs \
  --wiring-runs wiring_search_bool_sweep_v1 wiring_search_bool_seeds12_v1 wiring_search_digits_v2 wiring_search_mnist_smoke_v1 \
  --sequential-run sequential_register_required_tasks_v1 \
  --redundancy-runs redundant_hardening_bool_sweep_v1 redundant_hardening_bool_seeds12_v1 redundant_truth_refit_bool_seeds012_v1 redundant_hardening_digits_v1 redundant_hardening_mnist_smoke_v1 \
  --out-dir runs/reports_next_stage_v9
```

This prior report adds
`truth_refit_final_test_comparison.csv`, which compares the forced
`redundant_truth_refit_only` held-out test result against validation-selected
`redundant_task_hardened` and the strongest non-task-aware redundant baseline
for each covered Boolean dataset/seed pair. It is superseded by v10, which adds
digits and MNIST-smoke truth-refit-only final-test rows.

Prior next-stage evidence tables with Boolean + image-path
truth-table-refit-only final-test rows:

```bash
python build_next_stage_tables.py \
  --runs-root runs \
  --wiring-runs wiring_search_bool_sweep_v1 wiring_search_bool_seeds12_v1 wiring_search_digits_v2 wiring_search_mnist_smoke_v1 \
  --sequential-run sequential_register_required_tasks_v1 \
  --redundancy-runs redundant_hardening_bool_sweep_v1 redundant_hardening_bool_seeds12_v1 redundant_truth_refit_bool_seeds012_v1 redundant_hardening_digits_v1 redundant_truth_refit_digits_v1 redundant_hardening_mnist_smoke_v1 redundant_truth_refit_mnist_smoke_v1 \
  --out-dir runs/reports_next_stage_v10
```

This report extends
`truth_refit_final_test_comparison.csv` to include sklearn digits seed0 and
small binarized-MNIST seed0 truth-refit-only final-test rows. Image-path
coverage remains seed-0/smoke scale; full MNIST-scale validation,
image-dataset multi-seed validation, best-of-32, ABC, and learnable
connectivity remain incomplete or optional missing evidence.

Prior next-stage evidence tables with CIFAR-10 small smoke coverage:

```bash
python build_next_stage_tables.py \
  --runs-root runs \
  --wiring-runs wiring_search_bool_sweep_v1 wiring_search_bool_seeds12_v1 wiring_search_digits_v2 wiring_search_mnist_smoke_v1 \
  --sequential-run sequential_register_required_tasks_v1 \
  --redundancy-runs redundant_hardening_bool_sweep_v1 redundant_hardening_bool_seeds12_v1 redundant_truth_refit_bool_seeds012_v1 redundant_hardening_digits_v1 redundant_truth_refit_digits_v1 redundant_hardening_mnist_smoke_v1 redundant_truth_refit_mnist_smoke_v1 redundant_hardening_cifar10_smoke_v1 \
  --out-dir runs/reports_next_stage_v11
```

This prior report adds CIFAR-10 small smoke coverage to the
redundancy/truth-refit final-test audit. CIFAR remains small-budget path
coverage, not a full Mind-the-Gap CIFAR-scale comparison. It is superseded by
v12, which integrates the direct Mind-the-Gap/Gumbel comparison table while
keeping traditional full gap and method-native path gap separate.

Prior next-stage evidence tables with direct Mind-the-Gap/Gumbel comparison:

```bash
python build_next_stage_tables.py \
  --runs-root runs \
  --wiring-runs wiring_search_bool_sweep_v1 wiring_search_bool_seeds12_v1 wiring_search_digits_v2 wiring_search_mnist_smoke_v1 \
  --sequential-run sequential_register_required_tasks_v1 \
  --redundancy-runs redundant_hardening_bool_sweep_v1 redundant_hardening_bool_seeds12_v1 redundant_truth_refit_bool_seeds012_v1 redundant_hardening_digits_v1 redundant_truth_refit_digits_v1 redundant_hardening_mnist_smoke_v1 redundant_truth_refit_mnist_smoke_v1 redundant_hardening_cifar10_smoke_v1 \
  --mind-gap-runs bool_seeds012_timing_v3 digits_seeds012_timing_v3 cifar10_small_seed0_timing_v3 \
  --tuned-gumbel-comparison gumbel_tuned_comparison_seed0_timing_v3/comparison.csv \
  --out-dir runs/reports_next_stage_v12
```

This prior report adds `mind_gap_direct_comparison.csv`, which reports
`block_hard_full_gap` for the traditional relaxed-full vs hard-full
Mind-the-Gap comparison and `block_hard_native_gap` for the method-native
hard-prefix path. The two metrics must not be merged: current evidence shows
the native path can be better aligned while the primary full gap is still weak
against Gumbel. It is superseded by v13, which integrates the optional ABC
synthesis structural baseline.

Prior next-stage evidence tables with direct Mind-the-Gap/Gumbel and ABC
synthesis coverage:

```bash
python build_next_stage_tables.py \
  --runs-root runs \
  --wiring-runs wiring_search_bool_sweep_v1 wiring_search_bool_seeds12_v1 wiring_search_digits_v2 wiring_search_mnist_smoke_v1 \
  --sequential-run sequential_register_required_tasks_v1 \
  --redundancy-runs redundant_hardening_bool_sweep_v1 redundant_hardening_bool_seeds12_v1 redundant_truth_refit_bool_seeds012_v1 redundant_hardening_digits_v1 redundant_truth_refit_digits_v1 redundant_hardening_mnist_smoke_v1 redundant_truth_refit_mnist_smoke_v1 redundant_hardening_cifar10_smoke_v1 \
  --mind-gap-runs bool_seeds012_timing_v3 digits_seeds012_timing_v3 cifar10_small_seed0_timing_v3 \
  --tuned-gumbel-comparison gumbel_tuned_comparison_seed0_timing_v3/comparison.csv \
  --synthesis-report-dirs reports/synthesis_timing_v3 \
  --out-dir runs/reports_next_stage_v13
```

This prior report adds `abc_synthesis_comparison.csv` from the existing ABC
timing-v3 structural summary. ABC is used only as a structural baseline in v13:
`abc_post_and` and `abc_post_lev` are measured, but optimized BLIF networks are
not evaluated for post-ABC accuracy/gap. It is superseded by v14, which adds
post-ABC BLIF evaluation.

Post-ABC BLIF evaluation for the remote 210 Boolean ABC run:

```bash
python evaluate_abc_blif.py \
  --run-dir runs/bool_seed0_timing_v3_abc \
  --out-dir runs/reports_post_abc_eval_v3 \
  --run-name bool_seed0_timing_v3_abc \
  --abc-path /home/spco/boolean_sat/abc/abc
```

Prior next-stage evidence tables with direct Mind-the-Gap/Gumbel, ABC
synthesis, and post-ABC BLIF evaluation:

```bash
python build_next_stage_tables.py \
  --runs-root runs \
  --wiring-runs wiring_search_bool_sweep_v1 wiring_search_bool_seeds12_v1 wiring_search_digits_v2 wiring_search_mnist_smoke_v1 \
  --sequential-run sequential_register_required_tasks_v1 \
  --redundancy-runs redundant_hardening_bool_sweep_v1 redundant_hardening_bool_seeds12_v1 redundant_truth_refit_bool_seeds012_v1 redundant_hardening_digits_v1 redundant_truth_refit_digits_v1 redundant_hardening_mnist_smoke_v1 redundant_truth_refit_mnist_smoke_v1 redundant_hardening_cifar10_smoke_v1 \
  --mind-gap-runs bool_seeds012_timing_v3 digits_seeds012_timing_v3 cifar10_small_seed0_timing_v3 \
  --tuned-gumbel-comparison gumbel_tuned_comparison_seed0_timing_v3/comparison.csv \
  --synthesis-report-dirs reports/synthesis_timing_v3 \
  --post-abc-eval-dirs reports_post_abc_eval_v3 \
  --out-dir runs/reports_next_stage_v14
```

This prior report keeps ABC structural metrics and post-ABC functional checks
together. It is superseded by v15, which expands the direct tuned-Gumbel
comparison from Boolean-only seed-0 rows to digits, binarized-MNIST smoke, and
CIFAR-10 smoke rows.

Current next-stage evidence tables with expanded tuned-Gumbel direct
comparison:

```bash
python build_next_stage_tables.py \
  --runs-root runs \
  --wiring-runs wiring_search_bool_sweep_v1 wiring_search_bool_seeds12_v1 wiring_search_digits_v2 wiring_search_mnist_smoke_v1 \
  --sequential-run sequential_register_required_tasks_v1 \
  --redundancy-runs redundant_hardening_bool_sweep_v1 redundant_hardening_bool_seeds12_v1 redundant_truth_refit_bool_seeds012_v1 redundant_hardening_digits_v1 redundant_truth_refit_digits_v1 redundant_hardening_mnist_smoke_v1 redundant_truth_refit_mnist_smoke_v1 redundant_hardening_cifar10_smoke_v1 \
  --mind-gap-runs bool_seeds012_timing_v3 digits_seeds012_timing_v3 cifar10_small_seed0_timing_v3 \
  --tuned-gumbel-comparisons gumbel_tuned_comparison_seed0_timing_v3/comparison.csv gumbel_tuned_comparison_seed0_digits_layerdiag_v2/comparison.csv gumbel_tuned_comparison_seed0_mnist_layerdiag_v2/comparison.csv gumbel_tuned_comparison_seed0_cifar_layerdiag_v2/comparison.csv \
  --synthesis-report-dirs reports/synthesis_timing_v3 \
  --post-abc-eval-dirs reports_post_abc_eval_v3 \
  --out-dir runs/reports_next_stage_v15
```

This is the preferred current evidence report. It keeps ABC structural and
post-ABC BLIF checks from v14, and adds tuned seed-0 Gumbel comparisons for
sklearn digits, binarized-MNIST smoke, and CIFAR-10 smoke. The direct table now
reports both gap competitiveness and `delta_hard_acc_vs_gumbel`, because low
Gumbel gap alone is not a hard-accuracy win.

Post-ABC BLIF evaluation with fanout/unused-node metrics:

```bash
python evaluate_abc_blif.py \
  --run-dir runs/bool_seed0_timing_v3_abc \
  --out-dir runs/reports_post_abc_eval_v4 \
  --run-name bool_seed0_timing_v3_abc \
  --abc-path /home/spco/boolean_sat/abc/abc
```

Current next-stage evidence tables with expanded tuned-Gumbel direct
comparison and post-ABC BLIF fanout/unused-node metrics:

```bash
python build_next_stage_tables.py \
  --runs-root runs \
  --wiring-runs wiring_search_bool_sweep_v1 wiring_search_bool_seeds12_v1 wiring_search_digits_v2 wiring_search_mnist_smoke_v1 \
  --sequential-run sequential_register_required_tasks_v1 \
  --redundancy-runs redundant_hardening_bool_sweep_v1 redundant_hardening_bool_seeds12_v1 redundant_truth_refit_bool_seeds012_v1 redundant_hardening_digits_v1 redundant_truth_refit_digits_v1 redundant_hardening_mnist_smoke_v1 redundant_truth_refit_mnist_smoke_v1 redundant_hardening_cifar10_smoke_v1 \
  --mind-gap-runs bool_seeds012_timing_v3 digits_seeds012_timing_v3 cifar10_small_seed0_timing_v3 \
  --tuned-gumbel-comparisons gumbel_tuned_comparison_seed0_timing_v3/comparison.csv gumbel_tuned_comparison_seed0_digits_layerdiag_v2/comparison.csv gumbel_tuned_comparison_seed0_mnist_layerdiag_v2/comparison.csv gumbel_tuned_comparison_seed0_cifar_layerdiag_v2/comparison.csv \
  --synthesis-report-dirs reports/synthesis_timing_v3 \
  --post-abc-eval-dirs reports_post_abc_eval_v4 \
  --out-dir runs/reports_next_stage_v16
```

This v16 report supersedes v15 for the optional synthesis baseline only. It
does not change the Mind-the-Gap split: primary full-gap competitiveness with
Gumbel must still be read from the `full_gap_*` columns, while method-native
path gap and hard-accuracy deltas remain separate diagnostics.

Current next-stage evidence tables with candidate-set learnable connectivity:

```bash
python build_next_stage_tables.py \
  --runs-root runs \
  --wiring-runs wiring_search_bool_sweep_v1 wiring_search_bool_seeds12_v1 wiring_search_digits_v2 wiring_search_mnist_smoke_v1 wiring_candidate_learnable_bool_seed0_v1 \
  --sequential-run sequential_register_required_tasks_v1 \
  --redundancy-runs redundant_hardening_bool_sweep_v1 redundant_hardening_bool_seeds12_v1 redundant_truth_refit_bool_seeds012_v1 redundant_hardening_digits_v1 redundant_truth_refit_digits_v1 redundant_hardening_mnist_smoke_v1 redundant_truth_refit_mnist_smoke_v1 redundant_hardening_cifar10_smoke_v1 \
  --mind-gap-runs bool_seeds012_timing_v3 digits_seeds012_timing_v3 cifar10_small_seed0_timing_v3 \
  --tuned-gumbel-comparisons gumbel_tuned_comparison_seed0_timing_v3/comparison.csv gumbel_tuned_comparison_seed0_digits_layerdiag_v2/comparison.csv gumbel_tuned_comparison_seed0_mnist_layerdiag_v2/comparison.csv gumbel_tuned_comparison_seed0_cifar_layerdiag_v2/comparison.csv \
  --synthesis-report-dirs reports/synthesis_timing_v3 \
  --post-abc-eval-dirs reports_post_abc_eval_v4 \
  --out-dir runs/reports_next_stage_v17
```

This v17 report supersedes v16 for Part A coverage. It adds the candidate-set
connectivity rows and a same-run success check against
`fixed_random_matched_budget`; the current evidence is mixed and should be read
as a comparator rather than a connectivity-learning advantage.

Current next-stage evidence tables with multi-seed sequential register tasks:

```bash
python build_next_stage_tables.py \
  --runs-root runs \
  --wiring-runs wiring_search_bool_sweep_v1 wiring_search_bool_seeds12_v1 wiring_search_digits_v2 wiring_search_mnist_smoke_v1 wiring_candidate_learnable_bool_seed0_v1 \
  --sequential-runs sequential_register_required_tasks_v2 \
  --redundancy-runs redundant_hardening_bool_sweep_v1 redundant_hardening_bool_seeds12_v1 redundant_truth_refit_bool_seeds012_v1 redundant_hardening_digits_v1 redundant_truth_refit_digits_v1 redundant_hardening_mnist_smoke_v1 redundant_truth_refit_mnist_smoke_v1 redundant_hardening_cifar10_smoke_v1 \
  --mind-gap-runs bool_seeds012_timing_v3 digits_seeds012_timing_v3 cifar10_small_seed0_timing_v3 \
  --tuned-gumbel-comparisons gumbel_tuned_comparison_seed0_timing_v3/comparison.csv gumbel_tuned_comparison_seed0_digits_layerdiag_v2/comparison.csv gumbel_tuned_comparison_seed0_mnist_layerdiag_v2/comparison.csv gumbel_tuned_comparison_seed0_cifar_layerdiag_v2/comparison.csv \
  --synthesis-report-dirs reports/synthesis_timing_v3 \
  --post-abc-eval-dirs reports_post_abc_eval_v4 \
  --out-dir runs/reports_next_stage_v18
```

This v18 report supersedes v17 for Part B coverage. It uses
`--sequential-runs` and reports register-task success per task/seed instead of
collapsing repeated task rows.

Boolean best-of-32 hardening candidate check:

```bash
python redundant_hardening.py \
  --datasets parity8 majority9 random_sparse10 \
  --methods redundant_task_hardened \
  --redundancy-factors 2 4 8 \
  --seeds 0 \
  --base-width 32 \
  --base-votes 8 \
  --layers 3 \
  --epochs 30 \
  --batch-size 256 \
  --eval-batch-size 2048 \
  --best-of-n 32 \
  --out-dir runs/redundant_hardening_bool_bestof32_seed0_v1
```

This run is a scoped Part C candidate-generation check. It keeps the Boolean
training budget aligned with `redundant_hardening_bool_sweep_v1` and only
expands the Gumbel hardening candidate set from best-of-8 to best-of-32 for the
task-aware redundant method.

Current next-stage evidence tables with best-of-32 candidate coverage:

```bash
python build_next_stage_tables.py \
  --runs-root runs \
  --wiring-runs wiring_search_bool_sweep_v1 wiring_search_bool_seeds12_v1 wiring_search_digits_v2 wiring_search_mnist_smoke_v1 wiring_candidate_learnable_bool_seed0_v1 \
  --sequential-runs sequential_register_required_tasks_v2 \
  --redundancy-runs redundant_hardening_bool_sweep_v1 redundant_hardening_bool_seeds12_v1 redundant_truth_refit_bool_seeds012_v1 redundant_hardening_bool_bestof32_seed0_v1 redundant_hardening_digits_v1 redundant_truth_refit_digits_v1 redundant_hardening_mnist_smoke_v1 redundant_truth_refit_mnist_smoke_v1 redundant_hardening_cifar10_smoke_v1 \
  --mind-gap-runs bool_seeds012_timing_v3 digits_seeds012_timing_v3 cifar10_small_seed0_timing_v3 \
  --tuned-gumbel-comparisons gumbel_tuned_comparison_seed0_timing_v3/comparison.csv gumbel_tuned_comparison_seed0_digits_layerdiag_v2/comparison.csv gumbel_tuned_comparison_seed0_mnist_layerdiag_v2/comparison.csv gumbel_tuned_comparison_seed0_cifar_layerdiag_v2/comparison.csv \
  --synthesis-report-dirs reports/synthesis_timing_v3 \
  --post-abc-eval-dirs reports_post_abc_eval_v4 \
  --out-dir runs/reports_next_stage_v19
```

This v19 report supersedes v18 for Part C candidate coverage only. It should
remove `best_of_32_optional` from the coverage-audit missing list when
`gumbel_sample_0..31` are present, but it does not by itself establish a
primary full-gap win over Mind-the-Gap/Gumbel.

Consolidated timing-v3 report rebuild for the remote 210 project layout:

```bash
python summarize_gap_report.py \
  --runs-root runs \
  --runs bool_seeds012_timing_v3 digits_seeds012_timing_v3 cifar10_small_seed0_timing_v3 \
  --tuned-comparison runs/gumbel_tuned_comparison_seed0_timing_v3/comparison.csv \
  --out-dir reports/timing_v3
```

This writes the original aggregate checks plus
`depth_gap_accumulation_checks.csv`,
`strongest_seed0_gumbel_success_checks.csv` and
`strongest_seed0_gumbel_claim_checks.csv`, which compare seed-0
`block_hard_refit` directly against the sweep-selected
`gumbel_st_best_discrete_acc` row. The depth report is written when a run has
`aggregate/layer_diagnostics_by_layer.csv`.

The downloaded local mirror uses `remote_runs` paths instead:

```bash
python summarize_gap_report.py \
  --runs-root remote_runs \
  --runs bool_seeds012_timing_v3 digits_seeds012_timing_v3 cifar10_small_seed0_timing_v3 bool_seed0_layerdiag_v1 \
  --tuned-comparison remote_runs/gumbel_tuned_comparison_seed0_timing_v3/comparison.csv \
  --out-dir remote_runs/reports_timing_v3
```

ABC synthesis report rebuild for a run created with `--abc-stats` on the
remote 210 project layout:

```bash
python summarize_synthesis.py \
  --run-dir runs/bool_seed0_timing_v3_abc \
  --out-dir reports/synthesis_timing_v3 \
  --run-name bool_seed0_timing_v3_abc
```

The downloaded local mirror uses `remote_runs` paths instead:

```bash
python summarize_synthesis.py \
  --run-dir remote_runs/bool_seed0_timing_v3_abc \
  --out-dir remote_runs/reports_synthesis_timing_v3 \
  --run-name bool_seed0_timing_v3_abc
```

Metric definitions:

- `soft_acc` / `soft_loss` evaluate all trained relaxed blocks in soft mode.
- `discrete_acc` / `discrete_loss` evaluate the matching hard network: argmax
  gates for DLGN-style methods and per-block refit gates for
  `block_hard_refit`.
- `path_*` metrics evaluate the method-native training path. For
  `block_hard_refit`, this means hard frozen prefixes plus the final relaxed
  block compared with the final refit hard network.
- For block-wise methods, `epochs_to_target` is conservative: it is the total
  block epochs if the final hard model reaches the dataset target, otherwise
  `-1`. End-to-end methods track the first epoch where discrete accuracy
  crosses the target.
- `time_to_target` is the wall-clock companion to `epochs_to_target`: seconds
  until the first discrete target hit for end-to-end methods. For block-wise
  methods it is conservative final train time when the final hard model reaches
  target, otherwise `-1`. Per-epoch rows also include `elapsed_time` for
  convergence plots.

The default metrics do not run synthesis. With `--abc-stats`, the optional ABC
path exports final hard Boolean networks to BLIF and runs `strash; dc2` for
structural pre/post stats only. It does not feed optimized networks back into
PyTorch for re-evaluated accuracy or gap.

LightLogic Goal 6 distilled K-expanded student seed-1/2 confirmation run:

```bash
python lightlogic_distill_expansion.py \
  --datasets majority9 random_sparse10 \
  --seeds 1 2 \
  --k-values 4 \
  --init-modes fixed \
  --alphas 0.0 0.5 \
  --distill-taus 2.0 \
  --distill-epochs 120 \
  --student-lr 0.05 \
  --student-temp 0.1 \
  --width 128 \
  --layers 4 \
  --epochs 120 \
  --batch-size 256 \
  --eval-batch-size 2048 \
  --lr 0.01 \
  --estimator sinusoidal \
  --init residual \
  --device cpu \
  --out-dir runs/lightlogic_distill_bool_seeds12_k4fixed_v1
```

This is not a full hyperparameter sweep. It checks whether the best seed-0
candidate family, K=4/fixed with CE-only vs KL-distilled thresholds, still
improves deployable hard accuracy on additional seeds.

LightLogic Goal 6/5 distilled K-expanded student report:

```bash
python build_lightlogic_distill_report.py \
  --runs-root runs \
  --runs lightlogic_distill_bool_seed0_v1 lightlogic_distill_bool_seeds12_k4fixed_v1 \
  --k-merged runs/lightlogic_k_report_goal34_v4/k_expansion_merged_results.csv \
  --lightlogic-merged runs/lightlogic_report_goal012_v5/lightlogic_merged_results.csv \
  --out-dir runs/lightlogic_distill_report_goal56_v2
```

This report distinguishes CE-only student threshold tuning (`alpha=0`) from
teacher-logit KL distillation (`alpha>0`). It also treats data-weighted K
rounding as a diagnostic in the current unconstrained per-entry formulation:
when each truth-table entry independently chooses its own integer `r_ab`,
weighted and uniform nearest rounding choose the same result for positive
empirical weights. A nonzero Goal 5 effect therefore requires an additional
coupled or task-aware constraint, not just replacing uniform local MSE with
data-weighted local MSE.

LightLogic Goal 6+ truth-table adapter Boolean confirmation run:

```bash
python lightlogic_distill_expansion.py \
  --datasets majority9 random_sparse10 \
  --seeds 0 1 2 \
  --k-values 4 \
  --init-modes fixed \
  --adapter-modes none truth_table_st \
  --alphas 0.0 0.5 \
  --distill-taus 2.0 \
  --distill-epochs 160 \
  --student-lr 0.05 \
  --student-temp 0.1 \
  --width 128 \
  --layers 4 \
  --epochs 120 \
  --batch-size 256 \
  --eval-batch-size 2048 \
  --lr 0.01 \
  --estimator sinusoidal \
  --init residual \
  --device cpu \
  --out-dir runs/lightlogic_distill_adapter_bool_k4fixed_seeds012_v1
```

LightLogic Goal 6+ truth-table adapter digits check:

```bash
python lightlogic_distill_expansion.py \
  --datasets digits \
  --seeds 0 \
  --k-values 4 \
  --init-modes fixed \
  --adapter-modes none truth_table_st \
  --alphas 0.0 0.5 \
  --distill-taus 2.0 \
  --distill-epochs 160 \
  --student-lr 0.05 \
  --student-temp 0.1 \
  --width 320 \
  --layers 4 \
  --epochs 150 \
  --batch-size 256 \
  --eval-batch-size 2048 \
  --lr 0.01 \
  --estimator sinusoidal \
  --init residual \
  --device cpu \
  --out-dir runs/lightlogic_distill_adapter_digits_seed0_k4fixed_v1
```

Combined Goal 6+ report:

```bash
python build_lightlogic_distill_report.py \
  --runs-root runs \
  --runs lightlogic_distill_adapter_bool_k4fixed_seeds012_v1 lightlogic_distill_adapter_digits_seed0_k4fixed_v1 \
  --k-merged runs/lightlogic_k_report_goal34_v4/k_expansion_merged_results.csv \
  --lightlogic-merged runs/lightlogic_report_goal012_v5/lightlogic_merged_results.csv \
  --out-dir runs/lightlogic_distill_adapter_report_goal6plus_v2
```

`adapter_mode=none` is the threshold-only student-distillation baseline.
`adapter_mode=truth_table_st` learns the K-expanded truth-table entries with a
straight-through rounded `r/K` forward pass, so final hard inference remains a
deployable quantized truth-table network.

Scaled Goal 6+ image-task checks:

```bash
python lightlogic_distill_expansion.py \
  --datasets binarized_mnist \
  --seeds 0 \
  --k-values 4 \
  --init-modes fixed \
  --adapter-modes none truth_table_st \
  --alphas 0.0 0.5 \
  --distill-taus 2.0 \
  --distill-epochs 120 \
  --student-lr 0.05 \
  --student-temp 0.1 \
  --width 640 \
  --layers 4 \
  --epochs 120 \
  --batch-size 512 \
  --eval-batch-size 2048 \
  --lr 0.01 \
  --estimator sinusoidal \
  --init residual \
  --threshold-levels 1 \
  --image-max-train 4000 \
  --image-max-test 1000 \
  --data-dir /home/spco/data \
  --device cuda \
  --out-dir runs/lightlogic_distill_adapter_mnist_seed0_w640_v1
```

```bash
python lightlogic_distill_expansion.py \
  --datasets cifar10 \
  --seeds 0 \
  --k-values 4 \
  --init-modes fixed \
  --adapter-modes none truth_table_st \
  --alphas 0.0 0.5 \
  --distill-taus 2.0 \
  --distill-epochs 120 \
  --student-lr 0.05 \
  --student-temp 0.1 \
  --width 1600 \
  --layers 4 \
  --epochs 120 \
  --batch-size 512 \
  --eval-batch-size 2048 \
  --lr 0.01 \
  --estimator sinusoidal \
  --init residual \
  --threshold-levels 1 \
  --image-max-train 4000 \
  --image-max-test 1000 \
  --data-dir /home/spco/data \
  --device cuda \
  --out-dir runs/lightlogic_distill_adapter_cifar10_seed0_w1600_v1
```

Scaled LightLogic/DLGN baselines for those image-task checks:

```bash
python lightlogic_experiments.py \
  --datasets binarized_mnist \
  --methods light_iwp light_iwp_st dlgn_op \
  --seeds 0 \
  --width 640 \
  --layers 4 \
  --epochs 120 \
  --batch-size 512 \
  --eval-batch-size 2048 \
  --lr 0.01 \
  --estimator sinusoidal \
  --init residual \
  --threshold-levels 1 \
  --image-max-train 4000 \
  --image-max-test 1000 \
  --data-dir /home/spco/data \
  --device cuda \
  --out-dir runs/lightlogic_baseline_mnist_seed0_w640_v1
```

```bash
python lightlogic_experiments.py \
  --datasets cifar10 \
  --methods light_iwp light_iwp_st dlgn_op \
  --seeds 0 \
  --width 1600 \
  --layers 4 \
  --epochs 120 \
  --batch-size 512 \
  --eval-batch-size 2048 \
  --lr 0.01 \
  --estimator sinusoidal \
  --init residual \
  --threshold-levels 1 \
  --image-max-train 4000 \
  --image-max-test 1000 \
  --data-dir /home/spco/data \
  --device cuda \
  --out-dir runs/lightlogic_baseline_cifar10_seed0_w1600_v1
```

Scaled combined report:

```bash
python build_lightlogic_report.py \
  --runs-root runs \
  --runs lightlogic_bool_seed0_goal012_sigmoid_v1 lightlogic_bool_seed0_goal012_sinusoidal_v1 lightlogic_digits_seed0_goal02_sinusoidal_v1 lightlogic_baseline_mnist_seed0_w640_v1 lightlogic_baseline_cifar10_seed0_w1600_v1 \
  --out-dir runs/lightlogic_report_scaled_baselines_v1

python build_lightlogic_distill_report.py \
  --runs-root runs \
  --runs lightlogic_distill_adapter_bool_k4fixed_seeds012_v1 lightlogic_distill_adapter_digits_seed0_k4fixed_v1 lightlogic_distill_adapter_mnist_seed0_w640_v1 lightlogic_distill_adapter_cifar10_seed0_w1600_v1 \
  --k-merged runs/lightlogic_k_report_goal34_v4/k_expansion_merged_results.csv \
  --lightlogic-merged runs/lightlogic_report_scaled_baselines_v1/lightlogic_merged_results.csv \
  --out-dir runs/lightlogic_distill_adapter_report_scaled_v2
```

The scaled report supports adapter effectiveness on MNIST-4000/1000 and
digits, but not yet on CIFAR-10-small: CIFAR gets a small adapter-vs-none
gain, while remaining below the same-scale LightLogic-ST and OP/DLGN hard
baselines.
