# Consolidated required metrics

This directory joins the Boolean, sklearn-digits, and CIFAR-10-small scaled
runs into the requested schema:

```text
method | dataset | soft_acc | discrete_acc | acc_gap | soft_loss |
discrete_loss | loss_gap | train_time | epochs_to_target |
unused_gate_ratio | gate_count | depth | fanout_max
```

`required_metrics_table.csv` is machine-readable, and
`required_metrics_table.md` is the rendered table. The provenance variant adds
the remote run path, seed count, target-hit count, and target-hit rate. Values
are seed means; `-1` means no seed reached the target.
