#!/usr/bin/env bash
set -euo pipefail

cd /home/spco/sow_linear/ViT-LGN_goal6plus

SESSION="cls_topk_mixer_round1_20260628"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "session ${SESSION} already running"
else
    tmux new-session -d -s "${SESSION}" "cd /home/spco/sow_linear/ViT-LGN_goal6plus && CUDA_VISIBLE_DEVICES=1 bash run_cls_topk_token_mixer_ablation_20260628.sh"
    echo "started ${SESSION}"
fi
tmux ls
