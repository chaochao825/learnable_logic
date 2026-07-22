# ABC synthesis and functional evaluation

This capture contains seed-0 Boolean networks exported to BLIF and optimized
with `/home/spco/boolean_sat/abc/abc` using `strash; dc2`.

`synthesis/` contains the source networks. `post_abc_eval/optimized_blif/`
contains the rewritten networks. `post_abc_eval/post_abc_eval.csv` evaluates
both forms on the exact dataset inputs and records accuracy, loss, node count,
depth, fanout, unreachable nodes, and ABC runtime.

All 21 optimized networks match their source hard networks on every retained
Boolean test input. No post-ABC accuracy or loss changed. This is an exhaustive
evaluation of the held-out test sets, not a separate formal-equivalence proof.
