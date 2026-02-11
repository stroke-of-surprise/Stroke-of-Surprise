#!/usr/bin/env python3
"""
Unified training script for Stroke of Surprise.

This script provides a single entry point for training illusion sketches
using any of the three painter types (Bezier, B-spline, Neural).

Usage:
    # Using YAML config file
    python scripts/train.py --config_path config_files/defaults/bezier_dual_phase.yaml
    
    # Override any config values via command line (use dot notation for nested fields)
    python scripts/train.py --config_path config.yaml --seed 42 --num_iter 1000
    python scripts/train.py --config_path config.yaml --data.caption_A "a cat" --data.caption_B "a dog"
    python scripts/train.py --config_path config.yaml --optim.learning_rate 0.5
    
    # Generate multiple samples
    python scripts/train.py --config_path config.yaml --num_of_samples 3
"""

# Suppress deprecation warnings from diffusers
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="diffusers")

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import random
import numpy as np
import torch
import pydiffvg
import pyrallis

from src.configs import TrainConfig
from src.training import train_from_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def setup_device(use_cpu: bool = False) -> torch.device:
    """Set up compute device."""
    use_gpu = not use_cpu
    if not torch.cuda.is_available():
        use_gpu = False
        print("CUDA not available, using CPU")
    
    device = torch.device("cuda" if use_gpu else "cpu")
    
    # Configure pydiffvg
    pydiffvg.set_use_gpu(use_gpu)
    pydiffvg.set_device(device)
    
    return device


@pyrallis.wrap()
def main(cfg: TrainConfig):
    """
    Main entry point.
    
    Uses pyrallis.wrap() to automatically parse config from:
    1. Default values in TrainConfig dataclass
    2. YAML file specified via --config_path
    3. Command line arguments (can override any field with --field.subfield syntax)
    
    Examples:
        python scripts/train.py --config_path config.yaml
        python scripts/train.py --config_path config.yaml --seed 42
        python scripts/train.py --config_path config.yaml --data.caption_A "a cat"
        python scripts/train.py --config_path config.yaml --optim.sds_guidance_scale 50
    """
    # Setup device
    setup_device(cfg.use_cpu)
    
    # Run training for multiple samples if requested
    parent_dir = cfg.log.output_dir
    base_run_name = cfg.log.run_name
    base_seed = cfg.seed
    
    sample_id = 1
    samples_completed = 0
    
    def is_sample_completed(exp_dir: Path, num_phases: int) -> bool:
        """Check if a sample directory has completed outputs (sketch_p1.svg and sketch_p2.svg or sketch_pN.svg)."""
        if not exp_dir.exists():
            return False
        # Check if sketch_p1.svg exists
        if not (exp_dir / "sketch_p1.svg").exists():
            return False
        # Check if at least sketch_p2.svg or final phase exists
        if num_phases >= 2:
            return (exp_dir / "sketch_p2.svg").exists()
        return (exp_dir / "sketch.svg").exists()
    
    while samples_completed < cfg.num_of_samples:
        print(f"\n{'='*60}")
        print(f"Running sample {samples_completed + 1} of {cfg.num_of_samples}")
        print(f"{'='*60}")
        
        # Set run name with sample ID
        cfg.log.run_name = f"{base_run_name}-{sample_id}"
        cfg.log.output_dir = parent_dir
        
        # Find next available sample_id (skip directories that already have completed outputs)
        exp_dir = cfg.log.exp_dir
        while is_sample_completed(exp_dir, cfg.num_phases):
            print(f"Sample {sample_id} already completed (has sketch_p1.svg, sketch_p2.svg), skipping to next...")
            sample_id += 1
            cfg.log.run_name = f"{base_run_name}-{sample_id}"
            exp_dir = cfg.log.exp_dir
        
        print(f"Using experiment directory: {Path(exp_dir).absolute()}")
        
        # Set seed for this sample (based on samples_completed to ensure reproducibility)
        sample_seed = base_seed + samples_completed
        set_seed(sample_seed)
        cfg.seed = sample_seed
        
        # Create directories
        os.makedirs(exp_dir, exist_ok=True)
        
        # Run training
        try:
            train_from_config(cfg)
            print(f"Sample {sample_id} completed successfully")
        except Exception as e:
            print(f"Error training sample {sample_id}: {e}")
            import traceback
            traceback.print_exc()
        
        # Always increment samples_completed (even on error) to avoid infinite loops
        samples_completed += 1
        sample_id += 1
    
    print(f"\n{'='*60}")
    print(f"Training complete! Ran {samples_completed} samples")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
