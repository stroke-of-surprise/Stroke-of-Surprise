"""
B-spline curve painter implementation.

This module provides a painter that uses B-spline curves for stroke representation,
leveraging the calligraph library's SmoothingBSpline and Scene.
"""

import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pydiffvg
import torch
import matplotlib.cm as cm

from ..configs.data_config import PhaseConfig
from .base_painter import BasePainter
from ..curves import Scene, SmoothingBSpline, bspline, diffvg_cfg, make_deriv_loss


class BsplinePainter(BasePainter):
    """
    Painter implementation using B-spline curves.
    
    Each stroke is represented as a B-spline curve with configurable degree,
    multiplicity, and number of control points.
    """

    def __init__(
        self,
        num_strokes: int,
        canvas_size: int = 512,
        device: torch.device = None,
        spline_degree: int = 5,
        multiplicity: int = 1,
        num_control_points: int = 8,
        width: float = 2.5,
        min_width: float = 1.0,
        max_width: float = 8.0,
        optimize_width: bool = False,
        init_mode: str = 'circular',
        init_strokes_svg: Optional[str] = None,
    ):
        """
        Initialize B-spline painter.
        
        Args:
            num_strokes: Total number of strokes.
            canvas_size: Size of the canvas (square).
            device: Torch device.
            spline_degree: Degree of the B-spline.
            multiplicity: Multiplicity for sharper turns.
            num_control_points: Number of key points per stroke.
            width: Initial stroke width.
            min_width: Minimum allowed stroke width.
            max_width: Maximum allowed stroke width.
            optimize_width: Whether to optimize stroke widths.
            init_mode: Initialization mode ('circular', 'grid', or 'random').
            init_strokes_svg: Optional SVG file to initialize strokes from.
        """
        super().__init__(num_strokes, canvas_size, device)
        
        self.spline_degree = spline_degree
        self.multiplicity = multiplicity
        self.num_control_points = num_control_points
        self.width = width
        self.min_width = min_width
        self.max_width = max_width
        self.optimize_width = optimize_width
        self.init_mode = init_mode
        
        # Internal storage
        self.shapes = []
        self.shape_groups = []
        self.points_vars = []
        self.width_vars = []
        self.color_vars = []
        self.color_params = []
        self.optimize_flag = []
        self.initial_points = []
        
        # Phase tracking
        self.stroke_phase_index = None
        self.phase_prefix_counts = None
        
        # B-spline scene
        self.bspline_scene = None
        
        # SVG initialization
        self.init_strokes_svg = init_strokes_svg
        
        # Counter for initialization
        self._stroke_counter = 0

    def init_strokes(
        self,
        target_image: Optional[torch.Tensor] = None,
        phases: Optional[List[PhaseConfig]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize stroke parameters.
        
        Args:
            target_image: Optional target image (not used for B-spline).
            phases: List of phase configurations for multi-phase training.
        """
        self.shapes = []
        self.shape_groups = []
        self.color_params = []
        self.initial_points = []
        self._stroke_counter = 0
        
        # Initialize phase tracking if phases are provided
        if phases and len(phases) >= 2:
            self._init_phase_tracking(phases)
        
        # Generate strokes
        if self.init_strokes_svg is not None:
            self._load_strokes_from_svg()
        else:
            self._generate_strokes()
        
        # Create B-spline scene
        self._create_scene()
        
        self.optimize_flag = [True for _ in range(len(self.shapes))]
        self.strokes_initialized = True

    def _init_phase_tracking(self, phases: List[PhaseConfig]) -> None:
        """Initialize phase tracking from phase configurations."""
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
        
        if boundaries and boundaries[-1][1] < self.num_strokes:
            start, _ = boundaries[-1]
            boundaries[-1] = (start, self.num_strokes)
        
        self.stroke_phase_index = [0 for _ in range(self.num_strokes)]
        for p_idx, (start, end) in enumerate(boundaries):
            for i in range(start, min(end, self.num_strokes)):
                self.stroke_phase_index[i] = p_idx
        
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
                
                if canvas_width != self.canvas_size or canvas_height != self.canvas_size:
                    shape.points[:, 0] = (shape.points[:, 0] / canvas_width) * self.canvas_size
                    shape.points[:, 1] = (shape.points[:, 1] / canvas_height) * self.canvas_size
                
                paths.append(shape)
        
        # Convert loaded paths to B-spline shapes
        for i, path in enumerate(paths[:self.num_strokes]):
            self.shapes.append(self._create_bspline_from_path(path))
        
        # Fill remaining with generated strokes
        remaining = self.num_strokes - len(self.shapes)
        for _ in range(remaining):
            self.shapes.append(self._create_bspline_stroke())

    def _generate_strokes(self) -> None:
        """Generate B-spline strokes."""
        for _ in range(self.num_strokes):
            self.shapes.append(self._create_bspline_stroke())

    def _create_bspline_stroke(self) -> SmoothingBSpline:
        """Create a single B-spline stroke."""
        if self.init_mode == 'circular':
            keypoints = self._init_circular()
        elif self.init_mode == 'grid':
            keypoints = self._init_grid()
        else:
            p0 = (0.3 + 0.4 * random.random(), 0.3 + 0.4 * random.random())
            keypoints = self._init_random_walk(p0)
        
        keypoints = np.array(keypoints, dtype=np.float32)
        
        # Add multiplicity to keypoints
        Pw = bspline.add_multiplicity(keypoints, self.multiplicity)
        
        # Add stroke width as the third column
        # Add small random variation to initial widths to help them diverge during optimization
        if self.optimize_width:
            # Initialize with small random variation (±10% of base width)
            width_noise = np.random.uniform(-0.1, 0.1, len(Pw))
            width_values = np.ones(len(Pw)) * self.width * (1.0 + width_noise)
            # Clamp to valid range
            width_values = np.clip(width_values, self.min_width, self.max_width)
        else:
            # No variation if not optimizing widths
            width_values = np.ones(len(Pw)) * self.width
        Pw = np.hstack([Pw, width_values.reshape(-1, 1)])
        
        # Create SmoothingBSpline path
        path = SmoothingBSpline(
            Pw[:, :2],
            stroke_width=(Pw[:, 2], True),
            degree=self.spline_degree,
            multiplicity=self.multiplicity,
            closed=False,
        )
        
        self._stroke_counter += 1
        return path

    def _create_bspline_from_path(self, path: pydiffvg.Path) -> SmoothingBSpline:
        """Convert a pydiffvg.Path to SmoothingBSpline."""
        points = path.points.detach().cpu().numpy()
        
        # Sample keypoints from the path
        if len(points) > self.num_control_points:
            indices = np.linspace(0, len(points) - 1, self.num_control_points, dtype=int)
            keypoints = points[indices]
        else:
            keypoints = points
        
        Pw = bspline.add_multiplicity(keypoints, self.multiplicity)
        width_values = np.ones(len(Pw)) * self.width
        Pw = np.hstack([Pw, width_values.reshape(-1, 1)])
        
        return SmoothingBSpline(
            Pw[:, :2],
            stroke_width=(Pw[:, 2], True),
            degree=self.spline_degree,
            multiplicity=self.multiplicity,
            closed=False,
        )

    def _init_random_walk(self, p0: Tuple[float, float]) -> List[List[float]]:
        """Random walk initialization."""
        self.initial_points.append(p0)
        keypoints = [[p0[0] * self.canvas_size, p0[1] * self.canvas_size]]
        
        radius = 0.05
        current = list(p0)
        for _ in range(self.num_control_points - 1):
            dx = radius * (random.random() - 0.5)
            dy = radius * (random.random() - 0.5)
            current = [
                max(0.1, min(0.9, current[0] + dx)),
                max(0.1, min(0.9, current[1] + dy))
            ]
            keypoints.append([current[0] * self.canvas_size, current[1] * self.canvas_size])
            self.initial_points.append(tuple(current))
        
        return keypoints

    def _init_circular(self) -> List[List[float]]:
        """Circular initialization."""
        center = np.array([self.canvas_size / 2, self.canvas_size / 2])
        base_radius = self.canvas_size * 0.25
        
        stroke_idx = self._stroke_counter
        angle_offset = 2 * np.pi * stroke_idx / self.num_strokes
        radius = base_radius * (0.7 + 0.5 * stroke_idx / max(1, self.num_strokes))
        
        keypoints = []
        angles = np.linspace(0, 2 * np.pi, self.num_control_points, endpoint=False) + angle_offset
        
        for angle in angles:
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            keypoints.append([x, y])
            self.initial_points.append((x / self.canvas_size, y / self.canvas_size))
        
        return keypoints

    def _init_grid(self) -> List[List[float]]:
        """Grid-based initialization."""
        margin = self.canvas_size * 0.15
        grid_size = int(np.ceil(np.sqrt(self.num_strokes)))
        
        stroke_idx = self._stroke_counter
        row = stroke_idx // grid_size
        col = stroke_idx % grid_size
        
        cx = margin + (self.canvas_size - 2 * margin) * (col + 0.5) / grid_size
        cy = margin + (self.canvas_size - 2 * margin) * (row + 0.5) / grid_size
        
        radius = (self.canvas_size - 2 * margin) / (grid_size * 4)
        angles = np.linspace(0, 2 * np.pi, self.num_control_points, endpoint=False)
        
        keypoints = []
        for angle in angles:
            x = cx + radius * np.cos(angle)
            y = cy + radius * np.sin(angle)
            keypoints.append([x, y])
            self.initial_points.append((x / self.canvas_size, y / self.canvas_size))
        
        return keypoints

    def _create_scene(self) -> None:
        """Create calligraph Scene for rendering."""
        diffvg_cfg.one_channel_is_alpha = True
        
        self.bspline_scene = Scene()
        for shape in self.shapes:
            self.color_params.append(None)
            self.bspline_scene.add_shapes(
                [shape],
                stroke_color=([1.0], True),
                fill_color=None,
                split_primitives=True
            )
        
        self.shape_groups = self.bspline_scene.groups

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
            Rendered image tensor of shape (H, W, 3).
        """
        background_image = np.ones((self.canvas_size, self.canvas_size), dtype=np.float32)
        diffvg_cfg.one_channel_is_alpha = True
        
        if num_strokes is not None and num_strokes < len(self.shapes):
            temp_scene = Scene()
            for i in range(num_strokes):
                temp_scene.add_shapes(
                    [self.shapes[i]],
                    stroke_color=([1.0], True),
                    fill_color=None,
                    split_primitives=True
                )
            img = temp_scene.render(background_image)
        else:
            img = self.bspline_scene.render(background_image)
        
        # Handle grayscale output
        if len(img.shape) == 2:
            img = img.unsqueeze(2).repeat(1, 1, 3)
        elif img.shape[2] == 1:
            img = img.repeat(1, 1, 3)
        
        if return_alpha:
            alpha = torch.ones((img.shape[0], img.shape[1], 1), device=img.device)
            return img, alpha
        
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
        # Composite with white background (already done in render for grayscale)
        opacity = torch.ones((img.shape[0], img.shape[1], 1), device=img.device)
        img = opacity * img[:, :, :3] + torch.ones(
            img.shape[0], img.shape[1], 3, device=img.device
        ) * (1 - opacity)
        img = img.unsqueeze(0).permute(0, 3, 1, 2)
        return img

    def get_points_parameters(self) -> List[torch.Tensor]:
        """Get stroke point parameters for optimization."""
        self.points_vars = self.bspline_scene.get_points(only_grad=False)
        for pts in self.points_vars:
            pts.requires_grad = True
        return self.points_vars

    def get_color_parameters(self) -> List[torch.Tensor]:
        """Get stroke color parameters (not used for B-spline grayscale)."""
        return []

    def get_width_parameters(self) -> List[torch.Tensor]:
        """Get stroke width parameters for optimization."""
        if not self.optimize_width:
            return []
        
        self.width_vars = self.bspline_scene.get_stroke_widths(only_grad=False)
        for w in self.width_vars:
            w.requires_grad = True
        return self.width_vars

    def compute_smoothing_loss(
        self,
        num_strokes: Optional[int] = None,
        deriv_order: int = 3,
        weight: float = 200.0,
    ) -> torch.Tensor:
        """
        Compute B-spline smoothing loss.
        
        Args:
            num_strokes: Number of strokes to compute loss for.
            deriv_order: Derivative order for smoothing.
            weight: Weight for the loss.
            
        Returns:
            Smoothing loss value.
        """
        if weight <= 0:
            return torch.tensor(0.0, device=self.device)
        
        if num_strokes is not None and num_strokes > 0:
            shapes = self.shapes[:num_strokes]
        else:
            shapes = self.shapes
        
        if len(shapes) == 0:
            return torch.tensor(0.0, device=self.device)
        
        deriv_loss_fn = make_deriv_loss(deriv_order, self.canvas_size)
        return weight * deriv_loss_fn(shapes)

    def save_svg(
        self,
        path: Union[str, Path],
        num_strokes: Optional[int] = None,
        save_groups: bool = False,
        phases: Optional[List[PhaseConfig]] = None,
    ) -> List[str]:
        """
        Save strokes to SVG file(s).
        
        Args:
            path: Output path.
            num_strokes: Number of strokes to save.
            save_groups: Whether to save phase-colored version.
            phases: Phase configurations for coloring.
            
        Returns:
            List of saved file paths.
        """
        path = str(path)
        if path.endswith('.svg'):
            base_path = path[:-4]
            output_dir = str(Path(path).parent)
            name = Path(path).stem
        else:
            base_path = path
            output_dir = str(Path(path).parent)
            name = Path(path).name
        
        os.makedirs(output_dir, exist_ok=True)
        
        if num_strokes is not None and num_strokes > 0:
            shapes = self.shapes[:num_strokes]
        else:
            shapes = self.shapes
            num_strokes = len(shapes)
        
        svg_files = []
        
        # Build primitives from shapes
        all_primitives = []
        all_groups = []
        primitive_idx = 0
        
        for i, shape in enumerate(shapes):
            primitives = shape.primitives
            
            for prim in primitives:
                stroke_width = prim.stroke_width
                if hasattr(stroke_width, 'shape') and len(stroke_width.shape) > 0 and stroke_width.numel() > 1:
                    scalar_width = stroke_width.mean().detach()
                else:
                    scalar_width = stroke_width.detach() if torch.is_tensor(stroke_width) else torch.tensor(stroke_width)
                
                new_prim = pydiffvg.Path(
                    num_control_points=prim.num_control_points.clone(),
                    points=prim.points.clone().detach(),
                    stroke_width=scalar_width,
                    is_closed=prim.is_closed
                )
                all_primitives.append(new_prim)
            
            num_prims = len(primitives)
            shape_ids = list(range(primitive_idx, primitive_idx + num_prims))
            primitive_idx += num_prims
            
            # Determine stroke color
            if save_groups and self.stroke_phase_index is not None and phases and len(phases) > 1:
                num_phases = len(phases)
                phase_idx = self.stroke_phase_index[i] if i < len(self.stroke_phase_index) else 0
                phase_idx = max(0, min(num_phases - 1, phase_idx))
                cmap = cm.get_cmap('tab10', 10)
                rgba = cmap.colors[phase_idx]
                stroke_color = torch.tensor([float(rgba[0]), float(rgba[1]), float(rgba[2]), 1.0])
            else:
                stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
            
            for sid in shape_ids:
                group = pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor([sid]),
                    fill_color=None,
                    stroke_color=stroke_color
                )
                all_groups.append(group)
        
        # Save main SVG
        main_path = f"{base_path}.svg"
        pydiffvg.save_svg(main_path, self.canvas_size, self.canvas_size, all_primitives, all_groups)
        svg_files.append(main_path)
        
        # Save group-colored version if requested
        if save_groups and phases and len(phases) > 1:
            # Rename main as group
            group_path = f"{base_path}_group.svg"
            shutil.move(main_path, group_path)
            svg_files[0] = group_path
            
            # Save black version
            all_groups_black = []
            for idx in range(len(all_primitives)):
                group = pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor([idx]),
                    fill_color=None,
                    stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0])
                )
                all_groups_black.append(group)
            
            pydiffvg.save_svg(main_path, self.canvas_size, self.canvas_size, all_primitives, all_groups_black)
            svg_files.append(main_path)
        
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
        with torch.no_grad():
            # Clamp stroke widths
            for shape in self.shapes:
                if hasattr(shape, 'param'):
                    w = shape.param("stroke_width")
                    if w is not None and torch.is_tensor(w):
                        w.data.clamp_(self.min_width, self.max_width)

    def _render_mask(self, start_idx: int, end_idx: int) -> torch.Tensor:
        """Render a binary mask for specific strokes."""
        if end_idx <= start_idx:
            return torch.zeros(self.canvas_size, self.canvas_size, device=self.device)
        
        background_image = np.ones((self.canvas_size, self.canvas_size), dtype=np.float32)
        diffvg_cfg.one_channel_is_alpha = True
        
        temp_scene = Scene()
        for i in range(start_idx, end_idx):
            temp_scene.add_shapes(
                [self.shapes[i]],
                stroke_color=([1.0], True),
                fill_color=None,
                split_primitives=True
            )
        
        img = temp_scene.render(background_image)
        
        # Convert to mask (inverted grayscale)
        if len(img.shape) == 2:
            mask = 1.0 - img
        else:
            mask = 1.0 - img.mean(dim=-1)
        
        return mask

    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for checkpointing."""
        state = super().state_dict()
        state.update({
            'spline_degree': self.spline_degree,
            'multiplicity': self.multiplicity,
            'num_control_points': self.num_control_points,
            'width': self.width,
            'optimize_width': self.optimize_width,
            'stroke_phase_index': self.stroke_phase_index,
            'phase_prefix_counts': self.phase_prefix_counts,
        })
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from dictionary."""
        super().load_state_dict(state_dict)
        
        self.spline_degree = state_dict.get('spline_degree', self.spline_degree)
        self.multiplicity = state_dict.get('multiplicity', self.multiplicity)
        self.num_control_points = state_dict.get('num_control_points', self.num_control_points)
        self.width = state_dict.get('width', self.width)
        self.optimize_width = state_dict.get('optimize_width', self.optimize_width)
        self.stroke_phase_index = state_dict.get('stroke_phase_index')
        self.phase_prefix_counts = state_dict.get('phase_prefix_counts')

    def save_strokes(self, path: Union[str, Path]) -> None:
        """
        Save all stroke data to a pickle file for later re-rendering.
        
        Saves:
        - Control points for each stroke
        - Per-point stroke widths (preserving varying widths)
        - Stroke order and phase assignments
        - B-spline parameters (degree, multiplicity)
        
        Args:
            path: Output path for the pickle file.
        """
        import pickle
        
        strokes_data = []
        for i, shape in enumerate(self.shapes):
            # Get control points and widths from the SmoothingBSpline
            points = None
            widths = None
            
            if hasattr(shape, 'param'):
                # Get points parameter
                pts = shape.param('points')
                if pts is not None and torch.is_tensor(pts):
                    points = pts.detach().cpu().numpy()
                
                # Get stroke_width parameter
                w = shape.param('stroke_width')
                if w is not None and torch.is_tensor(w):
                    widths = w.detach().cpu().numpy()
            
            if points is None or widths is None:
                raise RuntimeError(
                    f"Cannot extract control points or widths from stroke {i}. "
                    f"points={points is not None}, widths={widths is not None}"
                )
            
            stroke_data = {
                'index': i,
                'points': points,
                'widths': widths,
                'phase_index': self.stroke_phase_index[i] if self.stroke_phase_index else 0,
            }
            strokes_data.append(stroke_data)
        
        save_data = {
            'strokes': strokes_data,
            'canvas_size': self.canvas_size,
            'spline_degree': self.spline_degree,
            'multiplicity': self.multiplicity,
            'num_strokes': len(self.shapes),
            'stroke_phase_index': self.stroke_phase_index,
            'phase_prefix_counts': self.phase_prefix_counts,
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Saved {len(strokes_data)} strokes to {path}")

    @classmethod
    def load_strokes(
        cls,
        path: Union[str, Path],
        device: torch.device = None,
    ) -> 'BsplinePainter':
        """
        Load strokes from a pickle file and create a new BsplinePainter.
        
        Args:
            path: Path to the pickle file.
            device: Torch device for the painter.
            
        Returns:
            BsplinePainter instance with loaded strokes.
        """
        import pickle
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        painter = cls(
            num_strokes=data['num_strokes'],
            canvas_size=data['canvas_size'],
            device=device or torch.device('cpu'),
            spline_degree=data['spline_degree'],
            multiplicity=data['multiplicity'],
        )
        
        # Rebuild strokes from saved data
        painter.shapes = []
        painter.stroke_phase_index = data.get('stroke_phase_index')
        painter.phase_prefix_counts = data.get('phase_prefix_counts')
        
        for stroke_data in data['strokes']:
            points = stroke_data['points']
            widths = stroke_data['widths']
            
            # Create Pw matrix with widths
            if len(widths) == len(points):
                Pw = np.hstack([points[:, :2], widths.reshape(-1, 1)])
            else:
                Pw = np.hstack([points[:, :2], np.full((len(points), 1), painter.width)])
            
            # Create SmoothingBSpline
            shape = SmoothingBSpline(
                Pw[:, :2],
                stroke_width=(Pw[:, 2], True),
                degree=painter.spline_degree,
                multiplicity=painter.multiplicity,
                closed=False,
            )
            painter.shapes.append(shape)
        
        # Create scene
        painter._create_scene()
        painter.strokes_initialized = True
        
        print(f"Loaded {len(painter.shapes)} strokes from {path}")
        return painter

    def render_selected(
        self,
        stroke_indices: List[int] = None,
        colors: List[Tuple[float, float, float]] = None,
        background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> torch.Tensor:
        """
        Render selected strokes with custom colors.
        
        Args:
            stroke_indices: List of stroke indices to render. None = all strokes.
            colors: List of RGB colors for each selected stroke. None = black.
            background_color: RGB background color.
            
        Returns:
            Rendered image tensor of shape (H, W, 3).
        """
        if stroke_indices is None:
            stroke_indices = list(range(len(self.shapes)))
        
        if colors is None:
            colors = [(0.0, 0.0, 0.0)] * len(stroke_indices)
        
        # Create background
        bg = np.ones((self.canvas_size, self.canvas_size, 3), dtype=np.float32)
        bg[:, :, 0] = background_color[0]
        bg[:, :, 1] = background_color[1]
        bg[:, :, 2] = background_color[2]
        
        # Render each stroke with its color - use CPU for compositing
        result = torch.from_numpy(bg)
        
        for i, idx in enumerate(stroke_indices):
            if idx >= len(self.shapes):
                continue
            
            # Render single stroke as grayscale mask
            background_gray = np.ones((self.canvas_size, self.canvas_size), dtype=np.float32)
            diffvg_cfg.one_channel_is_alpha = True
            
            temp_scene = Scene()
            temp_scene.add_shapes(
                [self.shapes[idx]],
                stroke_color=([1.0], True),
                fill_color=None,
                split_primitives=True
            )
            mask = temp_scene.render(background_gray)
            
            # Move mask to CPU for compositing
            mask = mask.cpu()
            
            # Convert to alpha mask (inverted)
            if len(mask.shape) == 2:
                alpha = 1.0 - mask
            else:
                alpha = 1.0 - mask.mean(dim=-1)
            
            alpha = alpha.unsqueeze(-1)  # (H, W, 1)
            
            # Apply color
            color = torch.tensor(colors[i], dtype=torch.float32).view(1, 1, 3)
            result = result * (1 - alpha) + color * alpha
        
        return result.to(self.device)

    def render_to_png(
        self,
        path: Union[str, Path],
        stroke_indices: List[int] = None,
        colors: List[Tuple[float, float, float]] = None,
        background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        """
        Render selected strokes and save to PNG.
        
        Args:
            path: Output PNG path.
            stroke_indices: List of stroke indices to render. None = all strokes.
            colors: List of RGB colors for each selected stroke.
            background_color: RGB background color.
        """
        from PIL import Image
        
        img = self.render_selected(stroke_indices, colors, background_color)
        img_np = (img.detach().cpu().numpy() * 255).astype(np.uint8)
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        Image.fromarray(img_np, 'RGB').save(path)
        print(f"Saved render to {path}")

