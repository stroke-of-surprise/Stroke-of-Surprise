"""
Bezier curve painter implementation.

This module provides a painter that uses Bezier curves for stroke representation,
rendered via pydiffvg.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pydiffvg
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.cm as cm

from ..configs.data_config import PhaseConfig
from .base_painter import BasePainter


class BezierPainter(BasePainter):
    """
    Painter implementation using Bezier curves.
    
    Each stroke is represented as a Bezier curve with configurable number
    of segments and control points per segment.
    """

    def __init__(
        self,
        num_strokes: int,
        canvas_size: int = 512,
        device: torch.device = None,
        num_segments: int = 1,
        control_points_per_seg: int = 4,
        width: float = 2.5,
        optimize_color: bool = False,
        background_image: Optional[Any] = None,
        init_strokes_svg: Optional[str] = None,
    ):
        """
        Initialize Bezier painter.
        
        Args:
            num_strokes: Total number of strokes.
            canvas_size: Size of the canvas (square).
            device: Torch device.
            num_segments: Number of Bezier segments per stroke.
            control_points_per_seg: Number of control points per segment.
            width: Stroke width.
            optimize_color: Whether to optimize stroke colors.
            background_image: Optional background image.
            init_strokes_svg: Optional SVG file to initialize strokes from.
        """
        super().__init__(num_strokes, canvas_size, device)
        
        self.num_segments = num_segments
        self.control_points_per_seg = control_points_per_seg
        self.width = width
        self.optimize_color = optimize_color
        
        # Internal storage
        self.shapes = []
        self.shape_groups = []
        self.points_vars = []
        self.color_vars = []
        self.color_params = []
        self.optimize_flag = []
        self.initial_points = []
        
        # Phase tracking
        self.stroke_phase_index = None
        self.phase_prefix_counts = None
        
        # Background image
        self.background_image = self._process_background_image(background_image) if background_image is not None else None
        
        # SVG initialization
        self.init_strokes_svg = init_strokes_svg

    def init_strokes(
        self,
        target_image: Optional[torch.Tensor] = None,
        phases: Optional[List[Dict]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize stroke parameters.
        
        Args:
            target_image: Optional target image (not used for Bezier, kept for interface).
            phases: List of phase configurations for multi-phase training.
        """
        self.shapes = []
        self.shape_groups = []
        self.color_params = []
        self.initial_points = []
        
        # Initialize phase tracking if phases are provided
        if phases and len(phases) >= 2:
            self._init_phase_tracking(phases)
        
        # Load strokes from SVG or generate randomly
        if self.init_strokes_svg is not None:
            self._load_strokes_from_svg()
        else:
            self._generate_random_strokes()
        
        # Initialize shape groups and colors
        self._init_shape_groups()
        
        self.optimize_flag = [True for _ in range(len(self.shapes))]
        self.strokes_initialized = True

    def _init_phase_tracking(self, phases: List[PhaseConfig]) -> None:
        """Initialize phase tracking from phase configurations."""
        # Build per-phase stroke ranges
        boundaries = []
        prev_end = 0
        
        for idx, ph in enumerate(phases):
            split = ph.num_strokes
            if split is not None:
                end = int(split)
                end = max(prev_end, min(self.num_strokes, end))
            else:
                end = self.num_strokes if idx == len(phases) - 1 else prev_end
            boundaries.append((prev_end, end))
            prev_end = end
        
        # Ensure final phase covers all strokes
        if boundaries and boundaries[-1][1] < self.num_strokes:
            start, _ = boundaries[-1]
            boundaries[-1] = (start, self.num_strokes)
        
        # Assign strokes to phases
        self.stroke_phase_index = [0 for _ in range(self.num_strokes)]
        for p_idx, (start, end) in enumerate(boundaries):
            for i in range(start, min(end, self.num_strokes)):
                self.stroke_phase_index[i] = p_idx
        
        # Compute phase prefix counts
        self.phase_prefix_counts = [min(self.num_strokes, end) for (_, end) in boundaries]

    def _load_strokes_from_svg(self) -> None:
        """Load strokes from SVG file."""
        if not os.path.isfile(self.init_strokes_svg):
            raise FileNotFoundError(f"SVG file not found: {self.init_strokes_svg}")
        
        canvas_width, canvas_height, shapes, _ = pydiffvg.svg_to_scene(self.init_strokes_svg)
        
        paths = []
        for shape in shapes:
            if isinstance(shape, pydiffvg.Path):
                shape.points = shape.points.to(self.device)
                
                # Scale points to match canvas size
                if canvas_width != self.canvas_size or canvas_height != self.canvas_size:
                    shape.points[:, 0] = (shape.points[:, 0] / canvas_width) * self.canvas_size
                    shape.points[:, 1] = (shape.points[:, 1] / canvas_height) * self.canvas_size
                
                shape.stroke_width = torch.tensor(self.width, device=self.device)
                
                if hasattr(shape, 'num_control_points'):
                    shape.num_control_points = shape.num_control_points.to(self.device)
                
                if shape.points.shape[0] > 0:
                    first_point = (
                        shape.points[0, 0].item() / self.canvas_size,
                        shape.points[0, 1].item() / self.canvas_size
                    )
                    self.initial_points.append(first_point)
                
                paths.append(shape)
        
        num_svg_strokes = len(paths)
        
        if num_svg_strokes < self.num_strokes:
            # Use all SVG strokes + generate random for the rest
            self.shapes.extend(paths)
            for _ in range(self.num_strokes - num_svg_strokes):
                self.shapes.append(self._create_random_path())
        elif num_svg_strokes > self.num_strokes:
            # Take only first num_strokes
            self.shapes.extend(paths[:self.num_strokes])
        else:
            self.shapes.extend(paths)

    def _generate_random_strokes(self) -> None:
        """Generate random stroke paths."""
        for _ in range(self.num_strokes):
            self.shapes.append(self._create_random_path())

    def _create_random_path(self) -> pydiffvg.Path:
        """Create a random Bezier path."""
        points = []
        num_control_points = torch.zeros(self.num_segments, dtype=torch.int32) + (self.control_points_per_seg - 2)
        
        # Random initialization in center region
        center_min = 0.3
        center_max = 0.7
        p0 = (
            center_min + (center_max - center_min) * random.random(),
            center_min + (center_max - center_min) * random.random()
        )
        self.initial_points.append(p0)
        points.append(p0)
        
        for _ in range(self.num_segments):
            radius = 0.05
            for _ in range(self.control_points_per_seg - 1):
                p1 = (
                    p0[0] + radius * (random.random() - 0.5),
                    p0[1] + radius * (random.random() - 0.5)
                )
                points.append(p1)
                p0 = p1
        
        points = torch.tensor(points, device=self.device)
        points[:, 0] *= self.canvas_size
        points[:, 1] *= self.canvas_size
        
        path = pydiffvg.Path(
            num_control_points=num_control_points,
            points=points,
            stroke_width=torch.tensor(self.width),
            is_closed=False
        )
        
        return path

    def _init_shape_groups(self) -> None:
        """Initialize shape groups with colors."""
        for i in range(len(self.shapes)):
            if self.optimize_color:
                # Random initialization for color optimization
                color_init = torch.tensor([
                    0.1 + random.random() * 0.2,
                    0.1 + random.random() * 0.2,
                    0.1 + random.random() * 0.2,
                    1.0
                ], device=self.device)
                color_param = color_init.clone().detach().requires_grad_(True)
                self.color_params.append(color_param)
            else:
                color_param = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
                self.color_params.append(None)
            
            shape_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([i]),
                fill_color=None,
                stroke_color=color_param
            )
            self.shape_groups.append(shape_group)

    def _process_background_image(self, background_image: Any) -> torch.Tensor:
        """Process background image to RGBA tensor."""
        if isinstance(background_image, Image.Image):
            if background_image.mode != 'RGB':
                background_image = background_image.convert('RGB')
            background_image = background_image.resize(
                (self.canvas_size, self.canvas_size),
                Image.Resampling.LANCZOS
            )
            bg_np = np.array(background_image).astype(np.float32) / 255.0
            bg_tensor = torch.from_numpy(bg_np).to(self.device)
        elif isinstance(background_image, np.ndarray):
            bg_np = background_image.copy().astype(np.float32)
            if bg_np.max() > 1.0:
                bg_np = bg_np / 255.0
            bg_tensor = torch.from_numpy(bg_np).to(self.device)
        elif isinstance(background_image, torch.Tensor):
            bg_tensor = background_image.to(self.device)
        else:
            raise TypeError(f"Unsupported background_image type: {type(background_image)}")
        
        # Ensure (H, W, 3) shape
        if bg_tensor.dim() == 2:
            bg_tensor = bg_tensor.unsqueeze(2).repeat(1, 1, 3)
        elif bg_tensor.shape[2] == 1:
            bg_tensor = bg_tensor.repeat(1, 1, 3)
        elif bg_tensor.shape[2] == 4:
            return bg_tensor
        
        # Add alpha channel
        alpha = torch.ones((bg_tensor.shape[0], bg_tensor.shape[1], 1), device=self.device)
        bg_tensor = torch.cat([bg_tensor, alpha], dim=2)
        
        return bg_tensor

    def render(
        self,
        num_strokes: Optional[int] = None,
        return_alpha: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Render strokes to an image.
        
        Args:
            num_strokes: Number of strokes to render.
            return_alpha: Whether to return alpha channel separately.
            
        Returns:
            Rendered image tensor of shape (H, W, 3) or (H, W, C).
        """
        if num_strokes is not None:
            shapes = self.shapes[:num_strokes]
            shape_groups = self.shape_groups[:num_strokes]
            color_params = self.color_params[:num_strokes]
        else:
            shapes = self.shapes
            shape_groups = self.shape_groups
            color_params = self.color_params
        
        # Update colors if optimizing
        if self.optimize_color:
            for sg, cp in zip(shape_groups, color_params):
                if cp is not None:
                    sg.stroke_color = torch.clamp(cp, 0.0, 1.0)
        
        # Render
        _render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            self.canvas_size, self.canvas_size, shapes, shape_groups
        )
        img = _render(
            self.canvas_size, self.canvas_size,
            2, 2, 0,
            self.background_image,
            *scene_args
        )
        
        if return_alpha:
            return img[:, :, :3], img[:, :, 3:4]
        
        # Composite with white background if no background image
        if self.background_image is None:
            opacity = img[:, :, 3:4]
            img = opacity * img[:, :, :3] + torch.ones(
                img.shape[0], img.shape[1], 3, device=self.device
            ) * (1 - opacity)
        else:
            img = img[:, :, :3]
        
        return img

    def get_image(self, num_strokes: Optional[int] = None) -> torch.Tensor:
        """
        Get rendered image in NCHW format.
        
        Args:
            num_strokes: Number of strokes to render.
            
        Returns:
            Image tensor of shape (1, 3, H, W).
        """
        img = self.render(num_strokes)
        img = img.unsqueeze(0).permute(0, 3, 1, 2)
        return img

    def get_points_parameters(self) -> List[torch.Tensor]:
        """Get stroke point parameters for optimization."""
        self.points_vars = []
        for i, path in enumerate(self.shapes):
            if self.optimize_flag[i]:
                path.points.requires_grad = True
                self.points_vars.append(path.points)
        return self.points_vars

    def get_color_parameters(self) -> List[torch.Tensor]:
        """Get stroke color parameters for optimization."""
        self.color_vars = []
        if self.optimize_color:
            for i, cp in enumerate(self.color_params):
                if self.optimize_flag[i] and cp is not None:
                    cp.requires_grad = True
                    self.color_vars.append(cp)
        return self.color_vars

    def get_width_parameters(self) -> List[torch.Tensor]:
        """Get stroke width parameters (not optimized in Bezier)."""
        return []

    def save_svg(
        self,
        path: Union[str, Path],
        num_strokes: Optional[int] = None,
        save_groups: bool = False,
        phases: Optional[List[Dict]] = None,
    ) -> List[str]:
        """
        Save strokes to SVG file(s).
        
        Args:
            path: Output path (without extension for multiple files).
            num_strokes: Number of strokes to save.
            save_groups: Whether to save phase-colored version.
            phases: Phase configurations for coloring.
            
        Returns:
            List of saved file paths.
        """
        path = str(path)
        if path.endswith('.svg'):
            base_path = path[:-4]
        else:
            base_path = path
        
        if num_strokes is not None and num_strokes > 0:
            shapes = self.shapes[:num_strokes]
            shape_groups = self.shape_groups[:num_strokes]
        else:
            shapes = self.shapes
            shape_groups = self.shape_groups
        
        # Store original colors
        original_colors = [
            sg.stroke_color.detach().clone() if sg.stroke_color is not None else None
            for sg in shape_groups
        ]
        
        svg_files = []
        
        if save_groups and phases and len(phases) > 1:
            # Save with phase colors
            cmap = cm.get_cmap('tab10', 10)
            for i, sg in enumerate(shape_groups):
                if sg.stroke_color is not None and self.stroke_phase_index is not None:
                    phase_idx = self.stroke_phase_index[i] if i < len(self.stroke_phase_index) else 0
                    phase_idx = min(phase_idx, len(phases) - 1)
                    rgba = cmap.colors[phase_idx]
                    sg.stroke_color = torch.tensor(
                        [float(rgba[0]), float(rgba[1]), float(rgba[2]), 1.0],
                        device=self.device
                    )
            
            group_path = f"{base_path}_group.svg"
            pydiffvg.save_svg(
                group_path,
                self.canvas_size, self.canvas_size,
                shapes, shape_groups
            )
            svg_files.append(group_path)
        
        # Save black version
        for sg in shape_groups:
            if sg.stroke_color is not None:
                sg.stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
        
        black_path = f"{base_path}.svg"
        pydiffvg.save_svg(
            black_path,
            self.canvas_size, self.canvas_size,
            shapes, shape_groups
        )
        svg_files.append(black_path)
        
        # Restore original colors
        for sg, orig_color in zip(shape_groups, original_colors):
            if orig_color is not None:
                sg.stroke_color = orig_color
        
        return svg_files

    def get_phase_stroke_count(self, phase_idx: int, phases: List['PhaseConfig']) -> int:
        """Get cumulative stroke count for a phase."""        
        if self.phase_prefix_counts is None:
            phase = phases[phase_idx]
            return phase.num_strokes if phase.num_strokes is not None else self.num_strokes
        
        phase_idx = max(0, min(len(self.phase_prefix_counts) - 1, phase_idx))
        return self.phase_prefix_counts[phase_idx]

    def clamp_parameters(self) -> None:
        """Clamp parameters to valid ranges."""
        # Clamp points to canvas bounds
        for path in self.shapes:
            path.points.data.clamp_(0, self.canvas_size)
        
        # Clamp colors to [0, 1]
        if self.optimize_color:
            for cp in self.color_params:
                if cp is not None:
                    cp.data.clamp_(0, 1)

    def _render_mask(self, start_idx: int, end_idx: int) -> torch.Tensor:
        """Render a binary mask for specific strokes."""
        if end_idx <= start_idx:
            return torch.zeros(self.canvas_size, self.canvas_size, device=self.device)
        
        # Create subset shapes and shape groups
        shapes = [self.shapes[i] for i in range(start_idx, end_idx)]
        shape_groups = []
        
        for new_idx, orig_idx in enumerate(range(start_idx, end_idx)):
            orig_sg = self.shape_groups[orig_idx]
            stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
            new_sg = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([new_idx]),
                fill_color=None,
                stroke_color=stroke_color
            )
            shape_groups.append(new_sg)
        
        # Render
        _render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            self.canvas_size, self.canvas_size, shapes, shape_groups
        )
        img = _render(self.canvas_size, self.canvas_size, 2, 2, 0, None, *scene_args)
        
        return img[:, :, 3]  # Return alpha channel as mask

    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for checkpointing."""
        state = super().state_dict()
        state.update({
            'num_segments': self.num_segments,
            'control_points_per_seg': self.control_points_per_seg,
            'width': self.width,
            'optimize_color': self.optimize_color,
            'stroke_phase_index': self.stroke_phase_index,
            'phase_prefix_counts': self.phase_prefix_counts,
            'shapes': [(s.points.detach().cpu(), s.num_control_points.cpu()) for s in self.shapes],
            'color_params': [
                cp.detach().cpu() if cp is not None else None
                for cp in self.color_params
            ],
        })
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from dictionary."""
        super().load_state_dict(state_dict)
        
        self.num_segments = state_dict.get('num_segments', self.num_segments)
        self.control_points_per_seg = state_dict.get('control_points_per_seg', self.control_points_per_seg)
        self.width = state_dict.get('width', self.width)
        self.optimize_color = state_dict.get('optimize_color', self.optimize_color)
        self.stroke_phase_index = state_dict.get('stroke_phase_index')
        self.phase_prefix_counts = state_dict.get('phase_prefix_counts')
        
        # Restore shapes
        if 'shapes' in state_dict:
            self.shapes = []
            for points, num_control_points in state_dict['shapes']:
                path = pydiffvg.Path(
                    num_control_points=num_control_points.to(self.device),
                    points=points.to(self.device),
                    stroke_width=torch.tensor(self.width),
                    is_closed=False
                )
                self.shapes.append(path)
        
        # Restore colors
        if 'color_params' in state_dict:
            self.color_params = [
                cp.to(self.device) if cp is not None else None
                for cp in state_dict['color_params']
            ]
            self._init_shape_groups()

