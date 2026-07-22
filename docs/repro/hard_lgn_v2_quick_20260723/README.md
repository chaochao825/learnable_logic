# Hard-LGN v2 equal-budget smoke evidence

This is the first all-method code-path check for the 2026-07-23 implementation.
It ran on server 210, CPU, from `hard_lgn_benchmark.py` with raw SHA256
`2377a89ce610e6bcdf896369c138cf303d48a322112f939a5d0ac86c3dc50bb3`.

Command:

```bash
python hard_lgn_gap_proto/hard_lgn_benchmark.py \
  --quick \
  --block-total-epochs 20 \
  --methods dlgn dlgn_anneal gumbel_st gumbel_st_cage hard_st hard_st_cage \
            block_relaxed block_hard_refit block_hard_task_refit \
  --device cpu \
  --refit-candidate-topk 16 \
  --out-dir hard_lgn_v2_quick_balanced_all_20260723
```

The three tiny Boolean splits and one seed are insufficient for research
claims. The artifact exists to verify that every method uses the same topology,
the block methods consume exactly the same 20-epoch total budget, every output
table is populated, and the new refit diagnostics are serializable. The bundled
`difflogic_compat.json` independently confirms primitive compatibility.

Final rows are in `raw/hard_lgn_v2_quick_balanced_all_20260723/results.csv`.
The `*.partial.csv` files equal their final counterparts byte-for-byte because
the run completed normally.
