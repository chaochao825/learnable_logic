# Mind-the-Gap scaled Boolean sweep

This capture contains the primary matched comparison from server 210.

- Remote source commit: `d1a9354`
- Python: `/home/wangmeiqi/anaconda3/envs/convlogic/bin/python`
- Architecture: fixed-wiring width 128, depth 4, 512 gates
- Budget: 120 total epochs for every method
- Seeds: 0, 1, 2
- Datasets: `parity8`, `majority9`, `random_sparse10`
- Protocol: `--mind-gap-scaled`

`results.csv` is the per-seed required table. `per_epoch.csv` records
convergence trajectories. `block_diagnostics.csv` records fitting source,
truth-table error, candidate selection, and block timing. `layer_diagnostics.csv`
records depth-wise representation mismatch. The `aggregate/` directory contains
seed means, variability, and comparisons against DLGN and Gumbel-ST.

This is a scaled controlled experiment, not the paper's full CIFAR setting.
