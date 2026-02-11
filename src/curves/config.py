#!/usr/bin/env python3
"""
Device configuration for curve rendering.

Adapted from calligraph library:
https://github.com/colormotor/calligraph/blob/main/calligraph/config.py

Original © Daniel Berio (@colormotor) 2025
"""

import torch

# Configure device based on available hardware
if torch.cuda.is_available():
    device_name = 'cuda'
    diffvg_device_name = 'cuda'
    has_gpu = True
    torch_dtype = torch.float32
else:
    device_name = 'cpu'
    diffvg_device_name = 'cpu'
    has_gpu = False
    torch_dtype = torch.float32

device = torch.device(device_name)
diffvg_device = torch.device(diffvg_device_name)


def clear_memory():
    """Clear GPU memory cache."""
    if has_gpu:
        import gc
        torch.cuda.empty_cache()
        gc.collect()

