#!/bin/bash
# Example: Run Neural painter with illusion sketch
# Usage: bash scripts/run_neural.sh
#
# Override any config value using dot notation:
#   --data.caption_A "a face"
#   --data.caption_B "a vase"
#   --num_of_samples 3

cd "$(dirname "$0")/.."

python scripts/train.py \
    --config_path config_files/defaults/neural_2phase.yaml \
    --data.caption_A "a rabbit" \
    --data.caption_B "a elephant" \
    --num_of_samples 30 \
    --log.output_dir ./output/neural/rabbit+elephant