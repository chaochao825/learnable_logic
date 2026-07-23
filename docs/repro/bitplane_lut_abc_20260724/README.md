# Bit-plane LUT truth-table synthesis audit

All 18 hard payloads were exported to BLIF, checked against 257
deterministic random Boolean inputs, optimized with ABC `strash; dc2`,
mapped back to 4-input LUTs, and formally compared with ABC CEC.
GroupSum and argmax are excluded so this report isolates learned LUT
truth-table and wiring compressibility.

The raw truth-table component is exactly 5,120 bits at 160 votes and
10,240 bits at 320 votes: `2 blocks * votes * 16 bits/LUT`. Those values
do not include the four discrete source indices per LUT. The comparable
whole learned-network inputs to synthesis are therefore 17,920 and 35,840
bits after encoding both truth tables and their realized wiring.

| mode | state/votes | original LUT4 | support=4 | support<=2 | unique tables | dictionary/raw | strash AND | dc2 AND | AIG reduction | mapped LUT4 | LUT4 reduction | mapped levels | K4 payload bits | payload reduction | fanout max | CEC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| argmax | 672/160 | 320 | 76.46% | 10.52% | 218.7 | 118.33% | 1823.7 | 1128.0 | 38.14% | 308.7 | 3.54% | 3.0 | 15265 | 14.82% | 14 | 3/3 |
| argmax | 832/320 | 640 | 78.65% | 10.16% | 391.3 | 117.40% | 3898.7 | 2336.7 | 40.05% | 596.0 | 6.87% | 3.0 | 32106 | 10.42% | 21 | 3/3 |
| truth | 672/160 | 320 | 77.50% | 9.38% | 232.0 | 122.50% | 1843.3 | 1143.7 | 37.97% | 311.3 | 2.71% | 3.0 | 15436 | 13.86% | 15 | 3/3 |
| truth | 832/320 | 640 | 79.48% | 9.11% | 406.0 | 119.69% | 3904.7 | 2359.3 | 39.57% | 612.7 | 4.27% | 3.0 | 32789 | 8.51% | 17 | 3/3 |
| wiring | 672/160 | 320 | 77.50% | 9.38% | 232.0 | 122.50% | 1843.3 | 1143.7 | 37.97% | 311.3 | 2.71% | 3.0 | 15436 | 13.86% | 15 | 3/3 |
| wiring | 832/320 | 640 | 79.48% | 9.11% | 406.0 | 119.69% | 3904.7 | 2359.3 | 39.57% | 612.7 | 4.27% | 3.0 | 32789 | 8.51% | 17 | 3/3 |

## Interpretation

These truth tables are synthesizable but not strongly table-compressible.
Roughly 76%-79% depend on all four inputs, while only 9%-11% depend on
at most two. A per-model unique-table dictionary is 17%-22% larger than
the raw 16-bit tables. ABC removes about 38%-40% of the direct AIG AND
expansion, but that is a different representation. On the comparable
K=4 basis, whole-network mapping removes only 2.7%-6.9% of LUTs and
increases the mapped level count from two to three. Learned wiring plus
truth-table payload falls by about 9%-15%, depending on method and width.

`dictionary/raw` is a storage-only template dictionary estimate; it
does not imply gates with different source wires can share hardware.
`mapped LUT4 reduction` is the directly comparable whole-network
reduction after ABC and includes dead-logic removal and cross-gate
rewriting. AIG AND counts are a different cost basis from truth bits.
