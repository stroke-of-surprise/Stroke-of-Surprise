#!/bin/bash
# Example: Run B-spline painter with multi-phase illusion sketch
# Usage: bash scripts/run_bspline.sh
#
# Override any config value using dot notation:
#   --data.caption_A "a cat"
#   --data.caption_B "a dog"  
#   --optim.sds_guidance_scale 10.0
#   --num_of_samples 3
# --bspline.width_optim true

cd "$(dirname "$0")/.."

python scripts/train.py \
    --config_path config_files/defaults/bspline_2phase.yaml \
    --data.caption_A "a chicken" \
    --data.caption_B "a monkey" \
    --num_of_samples 50 \
    --log.output_dir ./output/bspline/chicken+monkey \
    --bspline.min_width 0.5 \
    --bspline.max_width 5.0