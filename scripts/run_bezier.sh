#!/bin/bash
# Example: Run Bezier painter with dual phase illusion sketch
# Usage: bash scripts/run_bezier.sh
#
# Override any config value using dot notation:
#   --data.caption_A "a cat"
#   --data.caption_B "a dog"
#   --optim.overlay_loss_type None, dot, dice
#   --optim.overlay_loss_weight 0.5
#   --num_of_samples 3

cd "$(dirname "$0")/.."

python scripts/train.py \
    --config_path config_files/defaults/bezier_2phase.yaml \
    --data.caption_A "a rabbit" \
    --data.caption_B "a horse" \
    --num_of_samples 30 \
    --log.output_dir ./output/bezier/rabbit+horse