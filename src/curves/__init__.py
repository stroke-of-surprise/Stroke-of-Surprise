"""
Curve utilities for B-spline and Bezier representations.

Adapted from calligraph library:
https://github.com/colormotor/calligraph

Original © Daniel Berio (@colormotor) 2025
"""

from . import bspline
from . import bezier
from . import geom
from .config import device, diffvg_device, has_gpu
from .diffvg_utils import Scene, SmoothingBSpline, cfg as diffvg_cfg
from .spline_losses import make_deriv_loss, make_bbox_loss, bending_loss

__all__ = [
    "bspline",
    "bezier",
    "geom",
    "device",
    "diffvg_device",
    "has_gpu",
    "Scene",
    "SmoothingBSpline",
    "diffvg_cfg",
    "make_deriv_loss",
    "make_bbox_loss",
    "bending_loss",
]

