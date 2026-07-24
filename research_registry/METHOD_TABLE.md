# Unified method table

Generated from the machine-readable research registry. Hard accuracy is
the primary outcome; gap is diagnostic and never sufficient by itself.

| method | status | architecture | state boundary | training | hard runtime | best recorded hard result | conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dlgn_2input | reference | fixed random two-input 16-function LGN | one Boolean bit per gate | relaxed softmax then argmax | bool gate tables | boolean3 61.28% on validation (baseline) | required bad-gap baseline; useful hard accuracy and gap must both be reported |
| dlgn_anneal | screen | DLGN with temperature and entropy schedule | one Boolean bit per gate | annealed relaxed training | bool gate tables | no compatible result | sometimes improves hard accuracy but not a universal gap/capacity solution |
| gumbel_st_lgn | reference | fixed-wiring 16-function LGN | hard Boolean forward | Gumbel-Softmax straight-through | bool gate tables | boolean3 54.08% on validation (baseline) | strong direct gap baseline; low gap with collapsed accuracy is not a win |
| block_hard_refit | screen | block-wise relaxed LGN with per-block truth-table replacement | hard Boolean frozen prefixes | block train then local truth-table fit | bool gate tables | boolean3 65.49% on validation (screen) | prevents depth-wise mismatch accumulation but does not fix unused gates or image accuracy |
| block_task_refit | screen | block-wise hard LGN with task-aware coordinate refit | hard Boolean frozen prefixes | task-aware local refit | bool gate tables | boolean3 58.46% on validation (screen) | smaller average Boolean-task gap than direct refit but lower hard accuracy |
| lightlogic_iwp_st | screen | two-input four-entry IWP truth tables | one Boolean bit per gate | straight-through IWP | bool gate tables | no compatible result | parameter-efficient and competitive on selected small tasks; conditional not universal |
| full_k_count | screen | K-input truth-table expansion with full popcount propagation | integer count levels | teacher expansion and calibration | bool LUT plus integer counters | binarized_mnist 90.37% on test (screen) | strong evidence that repeated one-bit thresholding is the primary information loss |
| direct_lut_b3 | screen | direct-trained three-input LUT network | one Boolean bit per LUT | end-to-end relaxed LUT then hard table | bool LUT tables | binarized_mnist 83.57% on test (screen) | best local-LUT evidence on matched binarized MNIST but cost rises sharply |
| bitstate_progressive | screen | persistent Boolean tokens with fixed wiring and progressive hardening | bool | soft warmup then hard-ST | bool/int reference | cifar10 27.88% on test (screen) | very small gap and few entropy-unused gates but hard CIFAR accuracy remains low |
| bitstate_count_token | screen | persistent Boolean tokens with learned integer global-token thresholds | bool | progressive hard-ST | bool/int reference | cifar10 22.00% on test (screen) | count threshold improves short-budget hard accuracy but only at the input global token |
| full_discrete_a8 | reference | Wmag7/A8 shift-add ViT with exact integer attention | A8 code plus integer exponent | QAT shadows with exact hard carrier | operator-audited bool/int | cifar10 76.10% on validation (reference) | strict 76.10% accuracy carrier and integer baseline; not the logic-native method because dense learned Wmag7 projections dominate capacity |
| scalelogic_d12 | rejected | d12/e384 Wmag7/A8 ScaleLogic ViT | A8 code plus integer exponent | QAT shadows with exact hard carrier | operator-audited prefix only | cifar10 71.62% on validation (reject) | more parameters and depth regress under the matched 50k budget |
| logic_tree_local | rejected | shared 3x3 seven-gate bit-sliced logic tree | A8 bitplanes | hard forward with soft derivative | bool/int payload | cifar10 75.76% on validation (reject) | hard LUTs stayed identity; measured change was surrogate or seed noise |
| fixed_hadamard_global | rejected | integer Walsh-Hadamard global mixer | A8 code plus exponent | QAT exact integer hard path | bool/int payload | no compatible result | all matched 1k variants underperform content-dependent Top-K |
| global_a8_lut_tree | rejected | six-stage A8-by-A8 global ROM tree | A8 code plus exponent | hard lookup with interpolated training surrogate | bool/int payload | no compatible result | 72 MiB payload is sample-inefficient and loses the matched 1k screen |
| attention_clean_ceiling | external_ceiling | real-valued ViT with hard XNOR Top-K selector | real-valued | long-horizon augmented training | contains real-valued operators | cifar10 78.93% on test (ceiling) | 78.93% CIFAR test shows training budget and inductive bias ceiling only |
| vit_lgn_combine_stage598 | screen | ConvTree hard patch plus donor ViT-LGN | binary patch then floating carrier | output-level hard-forward/full-soft-backward STE | contains real-valued operators | cifar10 36.20% on test (screen) | reproducible hard-patch baseline; full model is not no-real-value compliant |
| vit_lgn_width_expansion | unvalidated | multiplied embed/head/state/pass widths | binary patch bottleneck then floating carrier | midpoint selector surrogate and gradient-only residual | contains real-valued operators | no compatible result | mechanically reasonable width ablation but expands after an 18-bit patch bottleneck |
| bitstate_count_message_v0 | rejected | persistent Boolean tokens with multi-threshold Top-K count message | bool | soft warmup then hard-ST | operator-audited bool/int | cifar10 20.70% on validation (reject); test 21.10% | active count encoding changed 12.30% of messages but mean best-validation hard accuracy fell 0.05 pp; 92.98% of merge lanes remained state identity |
| bitplane_lut_argmax | candidate | 512 protected input planes plus A8-grouped 4-input LUT vote planes with learned candidate wiring | eight Boolean planes per state symbol | block-wise hard-ST then direct argmax | bool LUT/wiring plus integer GroupSum | sklearn_digits 92.59% on validation (promote); test 90.74% | current strongest logic-native digits candidate: doubling learned votes improves test hard accuracy by 4.44 pp mean with 3/3 validation wins and zero inactive vote planes |
| bitplane_lut_spatial_argmax | candidate | 24,576 protected CIFAR A8 bit planes plus class-vote 4-input LUT planes with fixed image-local candidate pools | eight Boolean planes per raw state symbol | block-wise hard-ST then direct argmax and hard freeze | bool LUT/wiring plus integer GroupSum | no compatible result | pre-registered CIFAR-100 width/depth screen; promotion requires hard validation gains rather than a gap-only result |
| bitplane_lut_sequence_mixed_argmax | screen | 512 protected context planes plus character-vote 4-input LUT planes with deterministic global candidate pools | eight Boolean planes per character ID | block-wise hard-ST then direct argmax and hard freeze | bool LUT/wiring plus integer GroupSum | tiny_shakespeare_char 25.82% on validation (baseline); test 25.59% | matched Tiny Shakespeare baseline reaches 25.82% validation hard accuracy; it is a rolling Boolean context model rather than a Transformer |
| bitplane_lut_sequence_causal_argmax | candidate | 512 protected context planes plus character-vote 4-input LUT planes with recent-history and same-class recurrent candidates | eight Boolean planes per character ID | block-wise hard-ST then direct argmax and exact hard-prefix continuation | bool LUT/wiring plus integer GroupSum | tiny_shakespeare_char 36.27% on validation (screen); test 35.43% | v64-d2 reaches 36.275% validation hard accuracy and improves 2.455 pp over v32-d2 at 2x gates; strict execution has zero floating tensors, but unused gates rise 1.442 pp and accuracy remains 1.940 pp below trigram |
| bitplane_lut_refit | rejected | 512 protected input planes plus A8-grouped 4-input LUT vote planes with learned candidate wiring | eight Boolean planes per state symbol | block-wise hard-ST then class-balanced empirical truth-table refit/freeze | bool LUT/wiring plus integer GroupSum | sklearn_digits 90.00% on validation (reject); test 89.63% | independent empirical truth refit lowers validation hard accuracy by 1.67 pp mean versus argmax across six paired width/seed cases; lower mismatch does not compensate for task-margin loss |
| bitplane_lut_wiring_refit | rejected | 512 protected input planes plus A8-grouped 4-input LUT vote planes with learned candidate wiring | eight Boolean planes per state symbol | block-wise hard-ST then greedy discrete wiring/truth refit/freeze | bool LUT/wiring plus integer GroupSum | sklearn_digits 90.00% on validation (reject); test 89.63% | training changes 65.27%-73.67% of wiring from default but coordinate-greedy post-refit changes 0%; results exactly match rejected truth refit |
| bitplane_lut_abc_dc2_k4 | screen | ABC strash/dc2 optimization and K=4 remapping of the trained two-block LUT/wiring network | Boolean raw input to Boolean vote output | post-training logic optimization with no learned parameters | ABC-mapped Boolean BLIF with CEC | sklearn_digits 92.59% CEC-equivalent; 640->582 K4 | truth tables are synthesizable but only weakly K4-compressible: 2.7%-6.9% fewer LUTs at depth 3; all 18 source/optimized networks pass CEC |

## Registered synthesis

Synthesis accuracy is inherited only after exact export replay and
formal source/optimized netlist equivalence. GroupSum/argmax scope
limitations remain explicit in the method and protocol rows.

| synthesis | source result | seed | hard acc | test hard | gates | reduction | depth | payload bits | payload reduction | fanout | runtime (s) | equivalence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| bitplane_argmax_abc_w672_s0 | bitplane_argmax_w672_s0 | 0 | 89.63% | 85.56% | 320->312 | 2.50% | 2->3 | 17920->15378 | 14.19% | 11 | 0.09327483712695539 | abc_cec_equivalent |
| bitplane_argmax_abc_w672_s1 | bitplane_argmax_w672_s1 | 1 | 87.04% | 85.56% | 320->313 | 2.19% | 2->3 | 17920->15502 | 13.49% | 11 | 0.08678160794079304 | abc_cec_equivalent |
| bitplane_argmax_abc_w672_s2 | bitplane_argmax_w672_s2 | 2 | 87.41% | 84.81% | 320->301 | 5.94% | 2->3 | 17920->14914 | 16.77% | 14 | 0.0848877418320626 | abc_cec_equivalent |
| bitplane_argmax_abc_w832_s0 | bitplane_argmax_w832_s0 | 0 | 92.59% | 90.74% | 640->582 | 9.06% | 2->3 | 35840->31402 | 12.38% | 18 | 0.1196939810179174 | abc_cec_equivalent |
| bitplane_argmax_abc_w832_s1 | bitplane_argmax_w832_s1 | 1 | 90.74% | 89.63% | 640->606 | 5.31% | 2->3 | 35840->32656 | 8.88% | 16 | 0.12629386293701828 | abc_cec_equivalent |
| bitplane_argmax_abc_w832_s2 | bitplane_argmax_w832_s2 | 2 | 91.11% | 88.89% | 640->600 | 6.25% | 2->3 | 35840->32260 | 9.99% | 21 | 0.1174476349260658 | abc_cec_equivalent |

## Registered results

`hard_acc` is measured on `selection_split`. The explicit test column
is never used by the promotion script.

| result | method | dataset | protocol | seed(s) | split | soft acc | hard acc | gap | soft loss | hard loss | loss gap | test hard | time (s) | unused | inactive | gates | depth | fanout | runtime | decision |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| bool_dlgn_mean | dlgn_2input | boolean3 | boolean_w128_d4_120e_s012 | 3 | validation | 82.10% | 61.28% | 27.78% |  |  |  |  |  | 93.14% |  | 512 | 4 |  | hard_payload_not_operator_audited | baseline |
| bool_gumbel_mean | gumbel_st_lgn | boolean3 | boolean_w128_d4_120e_s012 | 3 | validation | 47.64% | 54.08% | 6.47% |  |  |  |  |  | 99.07% |  | 512 | 4 |  | hard_payload_not_operator_audited | baseline |
| bool_block_refit_mean | block_hard_refit | boolean3 | boolean_w128_d4_120e_s012 | 3 | validation | 51.61% | 65.49% | 17.71% |  |  |  |  |  | 98.76% |  | 512 | 4 |  | operator_audited_bool_int | screen |
| bool_task_refit_mean | block_task_refit | boolean3 | boolean_w128_d4_120e_s012 | 3 | validation | 50.48% | 58.46% | 11.11% |  |  |  |  |  | 98.50% |  | 512 | 4 |  | operator_audited_bool_int | screen |
| direct_lut_b3_mnist | direct_lut_b3 | binarized_mnist | direct_lut_mnist_w800_d3_80e_s0 | 0 | test |  | 83.57% |  |  |  |  |  |  |  |  | 7200 | 3 |  | hard_payload_not_operator_audited | screen |
| full_k_mnist_k16 | full_k_count | binarized_mnist | full_k_mnist_k16_capture | 0 | test |  | 90.37% |  |  |  |  |  |  |  |  |  |  |  | hard_payload_not_operator_audited | screen |
| bitstate_progressive_w4096 | bitstate_progressive | cifar10 | bitstate_full_w4096_30e_s0 | 0 | test | 27.02% | 26.44% | 0.58% |  |  |  |  | 1994.731 | 0.06% |  | 14976 | 9 | 228 | bit_exact_bool_int_reference | screen |
| bitstate_progressive_w8192 | bitstate_progressive | cifar10 | bitstate_full_w8192_30e_s0 | 0 | test | 27.93% | 27.88% | 0.05% |  |  |  |  | 3161.143 | 0.03% |  | 27264 | 9 | 429 | bit_exact_bool_int_reference | screen |
| bitstate_count_token_short | bitstate_count_token | cifar10 | bitstate_count_token_10k_8e_s0 | 0 | test | 19.30% | 22.00% | 2.70% |  |  |  |  | 203.3 | 1.07% |  | 14976 | 9 | 228 | bit_exact_bool_int_reference | screen |
| full_discrete_d6_strict | full_discrete_a8 | cifar10 | full_discrete_d6e192_50k_s42 | 42 | validation |  | 75.30% |  |  |  |  |  | 6239.249 |  |  |  |  |  | operator_audited_bool_int | baseline |
| full_discrete_d12e192 | full_discrete_a8 | cifar10 | full_discrete_d12e192_50k_s42 | 42 | validation |  | 76.10% |  |  |  |  |  | 14838.189 |  |  |  |  |  | hard_checkpoint_not_full_audited | screen |
| scalelogic_local0 | scalelogic_d12 | cifar10 | scalelogic_d12e384_50k_s42 | 42 | validation |  | 70.80% |  |  |  |  |  | 20086.884 |  |  |  |  |  | operator_audited_prefix | reject |
| scalelogic_local4 | scalelogic_d12 | cifar10 | scalelogic_d12e384_50k_s42 | 42 | validation |  | 71.62% |  |  |  |  |  | 24227.729 |  |  |  |  |  | operator_audited_prefix | reject |
| logic_tree_local1 | logic_tree_local | cifar10 | full_discrete_logic_tree_50k_s42 | 42 | validation |  | 75.76% |  |  |  |  |  |  |  |  |  |  |  | hard_payload_not_full_executor | reject |
| attention_clean_200k | attention_clean_ceiling | cifar10 | attention_clean_aug_200k_s0 | 0 | test |  | 78.93% |  |  |  |  |  |  |  |  |  |  |  | contains_real_values | ceiling |
| combine_stage598 | vit_lgn_combine_stage598 | cifar10 | combine_stage598_30k_s_selected | 0 | test |  | 36.20% |  |  |  |  |  |  |  |  |  |  |  | contains_real_values | screen |
| count_message_majority_s0 | bitstate_progressive | cifar10 | bitstate_count_message_10k_8e_s012 | 0 | validation | 19.45% | 21.05% | 1.60% | 2.205183 | 2.168988 | 0.036196 | 20.90% | 207.972 | 1.03% | 0.12% | 14976 | 9 | 228 | operator_audited_bool_int | baseline |
| count_message_majority_s1 | bitstate_progressive | cifar10 | bitstate_count_message_10k_8e_s012 | 1 | validation | 19.45% | 19.25% | 0.20% | 2.193935 | 2.192635 | 0.001300 | 19.55% | 158.244 | 1.02% | 0.08% | 14976 | 9 | 228 | operator_audited_bool_int | baseline |
| count_message_majority_s2 | bitstate_progressive | cifar10 | bitstate_count_message_10k_8e_s012 | 2 | validation | 16.40% | 19.90% | 3.50% | 2.211194 | 2.175260 | 0.035934 | 20.55% | 208.294 | 0.79% | 0.09% | 14976 | 9 | 228 | operator_audited_bool_int | baseline |
| count_message_v0_s0 | bitstate_count_message_v0 | cifar10 | bitstate_count_message_10k_8e_s012 | 0 | validation | 20.05% | 20.70% | 0.65% | 2.191638 | 2.161308 | 0.030330 | 21.10% | 209.962 | 1.20% | 0.21% | 14976 | 9 | 228 | operator_audited_bool_int | reject |
| count_message_v0_s1 | bitstate_count_message_v0 | cifar10 | bitstate_count_message_10k_8e_s012 | 1 | validation | 17.45% | 19.30% | 1.85% | 2.209597 | 2.201113 | 0.008484 | 21.25% | 129.030 | 1.04% | 0.11% | 14976 | 9 | 228 | operator_audited_bool_int | reject |
| count_message_v0_s2 | bitstate_count_message_v0 | cifar10 | bitstate_count_message_10k_8e_s012 | 2 | validation | 18.40% | 20.05% | 1.65% | 2.231748 | 2.181233 | 0.050514 | 20.45% | 129.366 | 0.79% | 0.13% | 14976 | 9 | 228 | operator_audited_bool_int | reject |
| full_discrete_d12_strict | full_discrete_a8 | cifar10 | full_discrete_d12e192_strict_validation5000 | 42 | validation |  | 76.10% |  |  |  |  |  | 14838.189 |  |  |  | 12 | 768 | operator_audited_bool_int | reference |
| bitplane_argmax_w672_s0 | bitplane_lut_argmax | sklearn_digits | bitplane_lut_digits_scale_s012_v2 | 0 | validation | 87.41% | 89.63% | 2.22% | 1.0212278012876157 | 0.9807055592536926 | 0.04052224203392307 | 85.56% | 17.221263521991204 | 1.88% | 0.00% | 320 | 2 | 7 | operator_audited_bool_int | baseline |
| bitplane_argmax_w672_s1 | bitplane_lut_argmax | sklearn_digits | bitplane_lut_digits_scale_s012_v2 | 1 | validation | 87.41% | 87.04% | 0.37% | 1.0337840892650463 | 0.9827806353569031 | 0.051003453908143204 | 85.56% | 16.00183077098336 | 1.88% | 0.00% | 320 | 2 | 8 | operator_audited_bool_int | baseline |
| bitplane_argmax_w672_s2 | bitplane_lut_argmax | sklearn_digits | bitplane_lut_digits_scale_s012_v2 | 2 | validation | 85.93% | 87.41% | 1.48% | 1.039253291377315 | 1.0020948648452759 | 0.037158426532039046 | 84.81% | 16.189310968999052 | 2.50% | 0.00% | 320 | 2 | 10 | operator_audited_bool_int | baseline |
| bitplane_argmax_w832_s0 | bitplane_lut_argmax | sklearn_digits | bitplane_lut_digits_scale_s012_v2 | 0 | validation | 92.22% | 92.59% | 0.37% | 0.7069139268663195 | 0.654370903968811 | 0.05254302289750845 | 90.74% | 18.8051538419968 | 2.97% | 0.00% | 640 | 2 | 12 | operator_audited_bool_int | promote |
| bitplane_argmax_w832_s1 | bitplane_lut_argmax | sklearn_digits | bitplane_lut_digits_scale_s012_v2 | 1 | validation | 92.22% | 90.74% | 1.48% | 0.7489579942491319 | 0.6752309799194336 | 0.0737270143296983 | 89.63% | 18.989761892997194 | 1.56% | 0.00% | 640 | 2 | 10 | operator_audited_bool_int | promote |
| bitplane_argmax_w832_s2 | bitplane_lut_argmax | sklearn_digits | bitplane_lut_digits_scale_s012_v2 | 2 | validation | 91.48% | 91.11% | 0.37% | 0.7063685099283854 | 0.6624418497085571 | 0.04392666021982827 | 88.89% | 18.982158528000582 | 1.41% | 0.00% | 640 | 2 | 11 | operator_audited_bool_int | promote |
| bitplane_truth_w672_s0 | bitplane_lut_refit | sklearn_digits | bitplane_lut_digits_scale_s012_v2 | 0 | validation | 89.63% | 88.52% | 1.11% | 1.0119923909505208 | 0.9648955464363098 | 0.047096844514211034 | 87.41% | 15.869087978993775 | 1.56% | 0.00% | 320 | 2 | 7 | operator_audited_bool_int | reject |
| bitplane_truth_w672_s1 | bitplane_lut_refit | sklearn_digits | bitplane_lut_digits_scale_s012_v2 | 1 | validation | 88.15% | 87.41% | 0.74% | 1.0379731354890047 | 0.9977318644523621 | 0.040241271036642656 | 84.44% | 16.830524674005574 | 1.88% | 0.00% | 320 | 2 | 8 | operator_audited_bool_int | reject |
| bitplane_truth_w672_s2 | bitplane_lut_refit | sklearn_digits | bitplane_lut_digits_scale_s012_v2 | 2 | validation | 85.19% | 84.44% | 0.74% | 1.055351878978588 | 1.0064880847930908 | 0.048863794185497245 | 83.33% | 15.872776608011918 | 3.12% | 0.00% | 320 | 2 | 10 | operator_audited_bool_int | reject |
| bitplane_truth_w832_s0 | bitplane_lut_refit | sklearn_digits | bitplane_lut_digits_scale_s012_v2 | 0 | validation | 90.74% | 89.26% | 1.48% | 0.6998324358904803 | 0.6534595489501953 | 0.04637288694028496 | 87.41% | 19.042382020008517 | 2.03% | 0.00% | 640 | 2 | 12 | operator_audited_bool_int | reject |
| bitplane_truth_w832_s1 | bitplane_lut_refit | sklearn_digits | bitplane_lut_digits_scale_s012_v2 | 1 | validation | 90.00% | 90.00% | 0.00% | 0.7151326497395833 | 0.6638849973678589 | 0.05124765237172446 | 89.63% | 18.81210493601975 | 1.88% | 0.00% | 640 | 2 | 10 | operator_audited_bool_int | reject |
| bitplane_truth_w832_s2 | bitplane_lut_refit | sklearn_digits | bitplane_lut_digits_scale_s012_v2 | 2 | validation | 90.37% | 88.89% | 1.48% | 0.7854925084997106 | 0.6820296049118042 | 0.10346290358790644 | 87.78% | 19.3061144730018 | 2.34% | 0.00% | 640 | 2 | 11 | operator_audited_bool_int | reject |
| bitplane_wiring_w672_s0 | bitplane_lut_wiring_refit | sklearn_digits | bitplane_lut_digits_scale_s012_v2 | 0 | validation | 89.63% | 88.52% | 1.11% | 1.0119923909505208 | 0.9648955464363098 | 0.047096844514211034 | 87.41% | 16.060434979997808 | 1.56% | 0.00% | 320 | 2 | 7 | operator_audited_bool_int | reject |
| bitplane_wiring_w672_s1 | bitplane_lut_wiring_refit | sklearn_digits | bitplane_lut_digits_scale_s012_v2 | 1 | validation | 88.15% | 87.41% | 0.74% | 1.0379731354890047 | 0.9977318644523621 | 0.040241271036642656 | 84.44% | 16.745719155995175 | 1.88% | 0.00% | 320 | 2 | 8 | operator_audited_bool_int | reject |
| bitplane_wiring_w672_s2 | bitplane_lut_wiring_refit | sklearn_digits | bitplane_lut_digits_scale_s012_v2 | 2 | validation | 85.19% | 84.44% | 0.74% | 1.055351878978588 | 1.0064880847930908 | 0.048863794185497245 | 83.33% | 16.149620218988275 | 3.12% | 0.00% | 320 | 2 | 10 | operator_audited_bool_int | reject |
| bitplane_wiring_w832_s0 | bitplane_lut_wiring_refit | sklearn_digits | bitplane_lut_digits_scale_s012_v2 | 0 | validation | 90.74% | 89.26% | 1.48% | 0.6998324358904803 | 0.6534595489501953 | 0.04637288694028496 | 87.41% | 18.769757409987506 | 2.03% | 0.00% | 640 | 2 | 12 | operator_audited_bool_int | reject |
| bitplane_wiring_w832_s1 | bitplane_lut_wiring_refit | sklearn_digits | bitplane_lut_digits_scale_s012_v2 | 1 | validation | 90.00% | 90.00% | 0.00% | 0.7151326497395833 | 0.6638849973678589 | 0.05124765237172446 | 89.63% | 19.4274806700123 | 1.88% | 0.00% | 640 | 2 | 10 | operator_audited_bool_int | reject |
| bitplane_wiring_w832_s2 | bitplane_lut_wiring_refit | sklearn_digits | bitplane_lut_digits_scale_s012_v2 | 2 | validation | 90.37% | 88.89% | 1.48% | 0.7854925084997106 | 0.6820296049118042 | 0.10346290358790644 | 87.78% | 19.48818402600591 | 2.34% | 0.00% | 640 | 2 | 11 | operator_audited_bool_int | reject |
| bitplane_shakespeare_smoke_mixed_v32_d2_s0 | bitplane_lut_sequence_mixed_argmax | tiny_shakespeare_char | bitplane_lut_shakespeare_char_scale_s0_v1 | 0 | validation | 23.08% | 25.82% | 2.74% | 2.94233236579895 | 2.763379954147339 | 0.17895241165161124 | 25.59% | 122.6843662429601 | 2.21% | 19.42% | 4160 | 2 | 18 | operator_audited_bool_int | baseline |
| bitplane_shakespeare_smoke_causal_v32_d2_s0 | bitplane_lut_sequence_causal_argmax | tiny_shakespeare_char | bitplane_lut_shakespeare_char_scale_s0_v1 | 0 | validation | 25.91% | 28.99% | 3.08% | 2.902787559509277 | 2.6991745735168458 | 0.20361298599243138 | 28.50% | 123.80756862810813 | 1.68% | 17.88% | 4160 | 2 | 100 | operator_audited_bool_int | screen |
| bitplane_shakespeare_causal_v32_d2_s0 | bitplane_lut_sequence_causal_argmax | tiny_shakespeare_char | bitplane_lut_shakespeare_char_scale_s0_v1 | 0 | validation | 33.89% | 33.82% | 0.07% | 2.4671805248260497 | 2.445071348571777 | 0.022109176254272445 | 33.16% | 1270.9511692419765 | 2.57% | 18.99% | 4160 | 2 | 127 | operator_audited_bool_int | screen |
| bitplane_shakespeare_causal_v64_d2_s0 | bitplane_lut_sequence_causal_argmax | tiny_shakespeare_char | bitplane_lut_shakespeare_char_scale_s0_v1 | 0 | validation | 36.41% | 36.27% | 0.14% | 2.3073506427764894 | 2.309064136123657 | 0.0017134933471676383 | 35.43% | 2728.7207847289974 | 4.01% | 7.76% | 8320 | 2 | 182 | operator_audited_bool_int | screen |

## Capacity configurations

Payload size counts serialized hard tensors, not training shadows. Empty
cells mean that the historical branch did not preserve the measurement.

| method | config | input bits | direct preserved bits | state bits/token | parameters | hard payload bits | gates | depth | fanout max | capacity diagnosis |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dlgn_2input | w128_d4 |  |  | 128 |  |  | 512 | 4 |  | one output bit per gate; fixed random fan-in |
| dlgn_anneal | w128_d4 |  |  | 128 |  |  | 512 | 4 |  | same capacity as DLGN; schedule changes optimization only |
| gumbel_st_lgn | w128_d4 |  |  | 128 |  |  | 512 | 4 |  | same capacity as DLGN; Gumbel-ST changes the estimator only |
| block_hard_refit | w128_d4 |  |  | 128 |  |  | 512 | 4 |  | same topology as DLGN; refit changes tables not capacity |
| block_task_refit | w128_d4 |  |  | 128 |  |  | 512 | 4 |  | same topology as DLGN; task-aware refit changes selection only |
| lightlogic_iwp_st | w128_d4 |  |  | 128 |  |  | 512 | 4 |  | four truth parameters per two-input gate versus sixteen for OP |
| full_k_count | k16 |  |  | 640 |  |  |  |  |  | K16 exposes 17 count levels; exact width depends on captured expansion row |
| direct_lut_b3 | w800_d3 |  |  | 800 |  |  | 7200 | 3 |  | larger local truth tables improve basis at higher synthesis cost |
| bitstate_progressive | w4096_d2d2 | 192 | 192 | 4096 | 247424 | 36612384 | 14976 | 9 | 228 | 192 raw thermometer bits retained; Top-K values collapse to one majority bit; 4,326,400 fixed XNOR and 4,259,840 value-count inputs/sample |
| bitstate_progressive | w8192_d2d2 | 192 | 192 | 8192 |  |  | 27264 | 9 | 429 | width doubles but literal ratio increases to 93.98 percent |
| bitstate_count_token | w4096_d2d2_count_global | 192 | 192 | 4096 |  |  | 14976 | 9 | 228 | learned count threshold only for the initial global token |
| full_discrete_a8 | d6e192 | 384 |  | 1536 | 3562752 |  |  | 6 | 768 | A8 state retains 256 levels per channel; dense fan-in max 768 and structural fanout upper bound 768 |
| full_discrete_a8 | d12e192 | 384 |  | 1536 | 7101696 | 681822688 |  | 12 | 768 | 76.10% full strict replay; schema-v6 audit payload stores redundant code/sign/plane/chunk forms; logical bit-packed weights are 56,712,192 bits; depth is not synthesized gate depth |
| scalelogic_d12 | d12e384_local0 | 384 |  | 3072 | 28359168 |  |  | 12 | 1536 | 4x parameters versus d6 without matched-budget accuracy scaling; dense structural fanout upper bound 1536 |
| scalelogic_d12 | d12e384_local4 | 384 |  | 3072 | 28372992 |  |  | 12 | 1536 | early depthwise branch adds local bias but only 0.54 pp over local0; dense structural fanout upper bound 1536 |
| logic_tree_local | d6e192_local1 | 384 |  | 1536 |  |  |  | 6 | 768 | seven two-input LUTs per channel/bitplane; all selected identity in the measured run |
| fixed_hadamard_global | d6e192_hadamard | 384 |  | 1536 |  |  |  | 6 | 768 | parameter-free global butterfly does not add learned capacity |
| global_a8_lut_tree | d12e384 | 384 |  | 3072 |  | 603979776 |  | 12 | 1536 | 72 MiB hard ROM payload; sparse coverage; dense structural fanout upper bound 1536 |
| attention_clean_ceiling | k8_augmented | 384 |  | 6144 |  |  |  | 6 | 768 | real-valued carrier; included only as an optimization and accuracy ceiling |
| vit_lgn_combine_stage598 | stage598 | 144 | 18 | 192 |  |  |  |  |  | ConvTree halves 144 to 18 before expansion; full model contains real-valued state |
| vit_lgn_width_expansion | width_multiplier2 | 144 | 18 | 384 |  |  |  |  |  | post-bottleneck width expansion and x concatenation do not add source information |
| bitstate_count_message_v0 | w4096_d2d2_count_message | 192 | 192 | 4096 | 247424 | 36614432 | 14976 | 9 | 228 | five encoded count levels; 133,120 threshold outputs/sample; 4,326,400 fixed XNOR and 4,259,840 value-count inputs/sample |
| bitplane_lut_argmax | w672_p512_v160_b2_k4_c16 | 512 | 512 | 672 | 25600 | 24608 | 320 | 2 | 10 | 512 raw planes are protected wires; 160 vote planes are learned per block; max observed three-seed payload fanout |
| bitplane_lut_argmax | w832_p512_v320_b2_k4_c16 | 512 | 512 | 832 | 51200 | 44608 | 640 | 2 | 12 | exact 2x learned gate/table scale versus v160; max observed three-seed payload fanout |
| bitplane_lut_refit | w672_p512_v160_b2_k4_c16 | 512 | 512 | 672 | 25600 | 24608 | 320 | 2 | 10 | same capacity as argmax; class-balanced empirical truth bits are refitted before freeze |
| bitplane_lut_refit | w832_p512_v320_b2_k4_c16 | 512 | 512 | 832 | 51200 | 44608 | 640 | 2 | 12 | same capacity as argmax; scaling must improve hard accuracy rather than only state entropy |
| bitplane_lut_wiring_refit | w672_p512_v160_b2_k4_c16 | 512 | 512 | 672 | 25600 | 24608 | 320 | 2 | 10 | same payload budget; coordinate search changes only registered source indices and truth bits |
| bitplane_lut_wiring_refit | w832_p512_v320_b2_k4_c16 | 512 | 512 | 832 | 51200 | 44608 | 640 | 2 | 12 | all learned deployment capacity remains LUT truth bits and discrete source indices |
| bitplane_lut_abc_dc2_k4 | w672_argmax_abc_k4_s012 | 512 | 512 | 672 | 25600 | 15502 | 313 | 3 | 14 | conservative max over three argmax seeds after ABC; excludes fixed GroupSum/argmax; source learned payload was 17920 bits |
| bitplane_lut_abc_dc2_k4 | w832_argmax_abc_k4_s012 | 512 | 512 | 832 | 51200 | 32656 | 606 | 3 | 21 | conservative max over three argmax seeds after ABC; excludes fixed GroupSum/argmax; source learned payload was 35840 bits |
| bitplane_lut_argmax | c100_mixed_v64_d2 | 24576 | 24576 | 30976 | 1024000 | 1482240 | 12800 | 2 |  | fixed mixed candidate baseline; each final vote has at most 16 raw-input support leaves before repeated-source reduction |
| bitplane_lut_spatial_argmax | c100_spatial_v64_d2 | 24576 | 24576 | 30976 | 1024000 | 1482240 | 12800 | 2 |  | same gate budget as mixed baseline; fixed 3x3 raw probes and same-class recurrent vote candidates |
| bitplane_lut_spatial_argmax | c100_spatial_v128_d2 | 24576 | 24576 | 37376 | 2048000 | 2698240 | 25600 | 2 |  | 2x votes and gates versus v64 at the same Boolean depth; final support remains bounded by 16 leaves |
| bitplane_lut_spatial_argmax | c100_spatial_v128_d4 | 24576 | 24576 | 37376 | 4096000 | 4746240 | 51200 | 4 |  | same vote width as v128-d2 with 2x depth/gates and at most 256 raw-input support leaves per final vote |
| bitplane_lut_spatial_argmax | c100_spatial_v256_d4 | 24576 | 24576 | 50176 | 8192000 | 9123840 | 102400 | 4 |  | 2x votes versus v128-d4; largest pre-registered logic-native capacity point |
| bitplane_lut_sequence_mixed_argmax | shakespeare_mixed_v32_d2 | 512 | 512 | 2592 | 332800 | 304128 | 4160 | 2 | 18 | matched global-routing baseline; 266240 learned LUT-plus-source bits and fixed Boolean encoder/integer readout metadata |
| bitplane_lut_sequence_causal_argmax | shakespeare_causal_v32_d2 | 512 | 512 | 2592 | 332800 | 304128 | 4160 | 2 | 127 | same gate and payload budget as mixed; candidates encode recent causal positions and same-class vote recurrence |
| bitplane_lut_sequence_causal_argmax | shakespeare_causal_v64_d2 | 512 | 512 | 4672 | 665600 | 636928 | 8320 | 2 | 182 | 2x vote width and hard gates at fixed depth; no embedding or learned dense numeric matrix |
| bitplane_lut_sequence_causal_argmax | shakespeare_causal_v64_d4 | 512 | 512 | 4672 | 1331200 | 1202688 | 16640 | 4 |  | 2x Boolean depth at fixed vote width; scale is accepted only with validation hard-accuracy improvement |
| bitplane_lut_sequence_causal_argmax | shakespeare_causal_v128_d4 | 512 | 512 | 8832 | 2662400 | 2533888 | 33280 | 4 |  | largest pre-registered character screen; learned payload is only truth bits and source indices |

## Mainline decision

`full_discrete_a8` remains the 76.10% strict integer accuracy reference,
but its dense learned Wmag7 projections make it an integer baseline, not
the logic-native mainline. `bitplane_lut_argmax` is the current
logic-native candidate: doubling learned vote planes from 160 to 320
improves mean validation hard accuracy by 3.46 pp and test hard accuracy
by 4.44 pp with 3/3 paired validation wins, zero inactive vote planes,
and a zero-real standalone runtime. Independent truth refit is rejected
at -1.67 pp mean validation hard accuracy versus argmax. The greedy
wiring refitter changes no post-training source even though training
itself moves 65.27%-73.67% of connections away from default routes.
ABC confirms all truth tables are logic-synthesizable, but the current
functions are only weakly K4-compressible: 2.7%-6.9% fewer mapped
LUTs, with mapped depth increasing from two to three. All 18 optimized
networks pass CEC; GroupSum and argmax are outside that synthesis scope.
Next work must use joint task-margin-aware hard fitting and matched
vote-width/spatial scaling without learned dense numeric matrices.
