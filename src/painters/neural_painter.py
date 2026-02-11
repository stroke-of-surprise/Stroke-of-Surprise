"""
Neural painter implementation for vector graphics generation.

This implementation merges NeuralSVG's model architecture:
- model.py (SketchModel) - checkpoint loading
- nerf_mlp_multi.py - NerfMLPMulti, embeddings
- painter_nerf.py - PainterNerf (mlp_pass)
- painter.py - Painter (initialization, rendering)

Follows BasePainter interface for compatibility with IllusionTrainer.
"""

import os
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pydiffvg
import termcolor
import torch
import torch.nn as nn
from transformers import get_scheduler
import webcolors
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.configs.optim_config import OptimConfig
from src.configs.log_config import LogConfig

from .base_painter import BasePainter
from ..configs.data_config import PhaseConfig
from ..utils import prepare_image, get_clip_attention_map, set_init_strokes_with_attention_map, tensor_to_pil
from ..utils.vis_utils import save_attention_maps

# Type aliases
ColorValue = Union[str, torch.Tensor]
ShapeList = List[Union[pydiffvg.Path, pydiffvg.Rect]]
ShapeGroupList = List[pydiffvg.ShapeGroup]


# =============================================================================
# Embedding Modules
# =============================================================================

class GaussianEmbedding(nn.Module):
    """Gaussian positional embedding for strokes."""

    def __init__(
        self,
        in_channels: int,
        num_shapes: int,
        emb_size: int,
        scale: int = 10,
        eps: float = 1e-4,
        skip_normalization: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_shapes = num_shapes
        self.funcs = [torch.sin, torch.cos]
        self.bvals = nn.Parameter(
            torch.normal(0, 1, (emb_size // 2, in_channels)) * scale
        )
        self.eps = eps
        self.skip_normalization = skip_normalization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.skip_normalization:
            x = self.normalize(x)
        x = x + self.eps
        out = []
        for func in self.funcs:
            out.append(func((2.0 * torch.pi * x) @ self.bvals.T))
        return torch.cat(out, -1)

    def normalize(self, shape_id: torch.Tensor) -> torch.Tensor:
        return 2 * (shape_id / (self.num_shapes - 1)) - 1


class GaussianEmbeddingRGB(nn.Module):
    """Gaussian embedding for RGB colors."""

    def __init__(self, in_channels: int, emb_size: int, scale: int = 10):
        super().__init__()
        if emb_size % 3 != 0:
            raise ValueError(f"emb_size must be divisible by 3, got {emb_size}")
        
        color_emb_size = emb_size // 3
        self.pe_red = GaussianEmbedding(in_channels, num_shapes=2, emb_size=color_emb_size, scale=scale, eps=1e-6)
        self.pe_green = GaussianEmbedding(in_channels, num_shapes=2, emb_size=color_emb_size, scale=scale, eps=1e-6)
        self.pe_blue = GaussianEmbedding(in_channels, num_shapes=2, emb_size=color_emb_size, scale=scale, eps=1e-6)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.shape[0] != 3:
            raise ValueError(f"Expected RGB input with shape (3, ...), got {rgb.shape}")
        enc_red = self.pe_red(rgb[0].unsqueeze(0))
        enc_green = self.pe_green(rgb[1].unsqueeze(0))
        enc_blue = self.pe_blue(rgb[2].unsqueeze(0))
        return torch.cat((enc_red, enc_green, enc_blue), -1)


# =============================================================================
# MLP Modules
# =============================================================================

class CustomMLP(nn.Module):
    """MLP with optional residual connections for color embedding."""

    def __init__(
        self,
        input_dim: int,
        intermediate_dim: int,
        output_dim: int,
        num_layers: int,
        color_embedding_dim: int = 0,
        use_residual_concat: bool = False,
    ):
        super().__init__()
        
        if use_residual_concat and color_embedding_dim == 0:
            raise ValueError(f"use_residual_concat requires color_embedding_dim > 0")
        
        self.use_residual_concat = use_residual_concat
        self.color_embedding_dim = color_embedding_dim if use_residual_concat else 0
        self.intermediate_dim = intermediate_dim
        self.intermediate_dim_extended = intermediate_dim + self.color_embedding_dim

        # First layer
        self.first_layer = nn.Sequential(
            nn.Linear(input_dim, self.intermediate_dim),
            nn.LayerNorm(self.intermediate_dim),
            nn.LeakyReLU(),
        )

        # Intermediate layers
        self.intermediate_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.intermediate_dim_extended, self.intermediate_dim),
                nn.LayerNorm(self.intermediate_dim),
                nn.LeakyReLU(),
            )
            for _ in range(num_layers - 1)
        ])

        # Output layer
        self.output_layer = nn.Linear(self.intermediate_dim_extended, output_dim)

    def forward(self, x: torch.Tensor, color_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.first_layer(x)
        for layer in self.intermediate_layers:
            if self.use_residual_concat and color_embedding is not None:
                x = torch.cat((x, color_embedding), dim=1)
            x = layer(x)
        if self.use_residual_concat and color_embedding is not None:
            x = torch.cat((x, color_embedding), dim=1)
        return self.output_layer(x)


class NerfMLPMulti(nn.Module):
    """Multi-output MLP for vector graphics generation."""

    def __init__(
        self,
        num_strokes: int,
        total_num_points: int,
        input_dim: int = 128,
        intermediate_dim: int = 128,
        num_layers: int = 2,
        use_nested_dropout: bool = False,
        use_color: bool = True,
        nested_dropout_probability: float = 0.5,
        truncation_start_idx: int = 4,
        points_prediction_scale: float = 0.1,
        device: str = "cuda",
        use_dropout_value: bool = False,
        dropout_emb_dim: int = 16,
        toggle_color: bool = False,
        toggle_color_input_dim: int = 12,
        toggle_color_bg_colors: Optional[List[str]] = None,
        color_name_to_value_map: Optional[Dict[str, torch.Tensor]] = None,
        toggle_color_method: str = "rgb",
        toggle_aspect_ratio: bool = False,
        toggle_aspect_ratio_values: Optional[List[str]] = None,
        aspect_ratio_emb_dim: Optional[int] = None,
    ):
        super().__init__()
        
        # Store config
        self.num_strokes = num_strokes
        self.total_num_points = total_num_points
        self.intermediate_dim = intermediate_dim
        self.num_layers = num_layers
        self.use_nested_dropout = use_nested_dropout
        self.nested_dropout_probability = nested_dropout_probability
        self.use_color = use_color
        self.device = device
        self.truncation_start_idx = truncation_start_idx
        self.points_prediction_scale = points_prediction_scale
        self.use_dropout_value = use_dropout_value
        self.toggle_color = toggle_color
        self.toggle_color_bg_colors = toggle_color_bg_colors or []
        self.color_name_to_value_map = color_name_to_value_map or {}
        self.toggle_color_method = toggle_color_method
        self.toggle_aspect_ratio = toggle_aspect_ratio
        self.toggle_aspect_ratio_values = toggle_aspect_ratio_values or ["1:1"]

        if num_layers < 1:
            raise ValueError(f"Invalid value num_layers={num_layers}")

        # Initialize dimensions
        self.dropout_emb_dim = dropout_emb_dim if use_dropout_value else 0
        self.aspect_ratio_dim = aspect_ratio_emb_dim if toggle_aspect_ratio else 0
        self.output_dim_points = total_num_points * 2
        self.output_dim_color = 3
        self.output_dim_opacity = 1
        
        # Handle color toggle dimensions
        if toggle_color:
            self.toggle_color_input_dim = toggle_color_input_dim
            if toggle_color_method == "rgb" and toggle_color_input_dim % 3 != 0:
                raise ValueError(f"toggle_color_input_dim must be divisible by 3")
        else:
            self.toggle_color_input_dim = 0

        # Set MLP input dimensions
        self.shape_input_dim = input_dim
        self.input_dim_mlp_points = self.shape_input_dim + self.dropout_emb_dim + self.aspect_ratio_dim
        self.input_dim_mlp_color = self.shape_input_dim + self.dropout_emb_dim + self.toggle_color_input_dim

        # Create embeddings
        self.pe = GaussianEmbedding(in_channels=1, num_shapes=num_strokes, emb_size=self.shape_input_dim)
        
        if use_dropout_value:
            self.dropout_pe = GaussianEmbedding(in_channels=1, num_shapes=num_strokes, emb_size=self.dropout_emb_dim)

        if toggle_aspect_ratio:
            self.aspect_ratio_pe = GaussianEmbedding(
                in_channels=1, num_shapes=0, emb_size=self.aspect_ratio_dim, skip_normalization=True
            )

        if toggle_color:
            if toggle_color_method == "discrete":
                self.toggle_color_pe = GaussianEmbedding(
                    in_channels=1, num_shapes=len(toggle_color_bg_colors), emb_size=toggle_color_input_dim
                )
            elif toggle_color_method == "rgb":
                self.toggle_color_pe = GaussianEmbeddingRGB(in_channels=1, emb_size=toggle_color_input_dim)
            else:
                raise ValueError(f"Invalid toggle_color_method: {toggle_color_method}")
            
            self.toggle_color_mlp = CustomMLP(
                toggle_color_input_dim, intermediate_dim, toggle_color_input_dim, num_layers=1
            )

        # Create MLPs
        self.mlp_points = CustomMLP(
            self.input_dim_mlp_points, intermediate_dim, self.output_dim_points, num_layers
        )
        
        if use_color:
            self.mlp_color = CustomMLP(
                self.input_dim_mlp_color, intermediate_dim, self.output_dim_color, num_layers,
                color_embedding_dim=self.toggle_color_input_dim,
                use_residual_concat=toggle_color,
            )

        # Additional parameters
        self.width_inputs = torch.ones((1, num_strokes)).to(device) * 1.5
        self.opacities_defaults = torch.ones((1, num_strokes)).to(device)
        self.widths_output_layer = None
        self.gate = None
        self.background_color_params = nn.Parameter(torch.rand(4))

    def forward(
        self,
        truncation_idx: Optional[int] = None,
        toggle_color_value: Optional[ColorValue] = None,
        toggle_aspect_ratio_value: Optional[str] = None,
        indices_to_pass: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through network."""
        number_of_shapes = truncation_idx if truncation_idx is not None else self.num_strokes
        indices_to_pass = indices_to_pass if indices_to_pass is not None else list(range(number_of_shapes))

        outputs_dict = self.forward_single_stroke(
            indices_to_pass,
            toggle_color_value=toggle_color_value,
            toggle_aspect_ratio_value=toggle_aspect_ratio_value,
        )
        
        # Process points
        points = outputs_dict["points"]
        bs, _ = points.shape
        strokes = points.reshape(bs, -1, 2).unsqueeze(0)
        strokes = self.points_prediction_scale * strokes

        # Process colors
        colors = None
        if self.use_color and "color" in outputs_dict:
            colors = nn.Sigmoid()(outputs_dict["color"].T)

        return strokes, self.width_inputs, self.opacities_defaults, colors

    def forward_single_stroke(
        self,
        indices_to_pass: List[int],
        toggle_color_value: Optional[ColorValue] = None,
        toggle_aspect_ratio_value: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for strokes."""
        # Get base positional encoding
        indices_b = torch.tensor(indices_to_pass, device=self.device).float().unsqueeze(-1)
        encoding = self.pe.forward(indices_b)

        if self.use_dropout_value:
            dropout_encoding = self.dropout_pe.forward(indices_b)
            encoding = torch.cat((encoding, dropout_encoding), dim=-1)

        stroke_encoding_mlp_points = encoding
        stroke_encoding_mlp_color = encoding
        color_embedding = None

        # Add aspect ratio encoding if needed
        if self.toggle_aspect_ratio and toggle_aspect_ratio_value is not None:
            ratio_float_value = torch.tensor(
                [calculate_ratio(toggle_aspect_ratio_value)], device=self.device
            ).unsqueeze(-1)
            aspect_ratio_encoding = self.aspect_ratio_pe.forward(ratio_float_value)
            aspect_ratio_encoding = aspect_ratio_encoding.expand(encoding.shape[0], -1)
            stroke_encoding_mlp_points = torch.cat((stroke_encoding_mlp_points, aspect_ratio_encoding), dim=1)

        # Add color encoding if needed
        if self.toggle_color and toggle_color_value is not None:
            color_encoding = self._get_color_encoding(toggle_color_value)
            color_embedding = self.toggle_color_mlp(color_encoding)
            color_embedding = color_embedding.expand(encoding.shape[0], -1)
            stroke_encoding_mlp_color = torch.cat((encoding, color_embedding), dim=1)

        # Generate outputs
        outputs = {"points": self.mlp_points(stroke_encoding_mlp_points)}
        if self.use_color:
            outputs["color"] = self.mlp_color(stroke_encoding_mlp_color, color_embedding=color_embedding)

        return outputs

    def _get_color_encoding(self, toggle_color_value: ColorValue) -> torch.Tensor:
        """Get color encoding based on method."""
        if self.toggle_color_method == "discrete":
            color_number = self.toggle_color_bg_colors.index(toggle_color_value)
            return self.toggle_color_pe.forward(torch.tensor([color_number], device=self.device).float())
        elif self.toggle_color_method == "rgb":
            if isinstance(toggle_color_value, str):
                rgba = self.color_name_to_value_map[toggle_color_value]
            elif isinstance(toggle_color_value, torch.Tensor):
                rgba = toggle_color_value
            else:
                raise ValueError(f"Invalid toggle_color_value type: {type(toggle_color_value)}")
            rgb = rgba[:3].to(device=self.device).unsqueeze(-1)
            return self.toggle_color_pe.forward(rgb)
        else:
            raise ValueError(f"Invalid toggle_color_method: {self.toggle_color_method}")

    def apply_nested_dropout(
        self,
        embedding: torch.Tensor,
        truncation_idx: Optional[int] = None,
        end_truncation_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Apply nested dropout to embedding."""
        if truncation_idx is None:
            return embedding

        if truncation_idx is not None and end_truncation_idx is not None:
            if truncation_idx > end_truncation_idx:
                raise ValueError(f"truncation_idx={truncation_idx}, end_truncation_idx={end_truncation_idx}")

        for idx in range(embedding.shape[0]):
            if end_truncation_idx is None:
                embedding[idx][truncation_idx:] = 0
            else:
                embedding[idx][truncation_idx:end_truncation_idx] = 0

        return embedding

    @staticmethod
    def sample_truncation_idx(
        vector_size: int = 16, 
        start_idx: int = 1, 
        sampling_method: str = "uniform"
    ) -> int:
        """Sample truncation index."""
        if sampling_method == "uniform":
            return random.randint(start_idx, vector_size)
        elif sampling_method == "exp_decay":
            dist = exponential_decay_distribution(
                size=vector_size - start_idx + 1, temperature=1.5, last_item_prob=0.5
            )
            return np.random.choice(
                np.arange(start_idx, vector_size + 1), size=1, p=dist
            ).item()
        else:
            raise ValueError(f"Invalid sampling_method: {sampling_method}")


def init_weights(m: nn.Module) -> None:
    """Initialize network weights."""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def exponential_decay_distribution(
    size: int, temperature: float = 1.5, last_item_prob: float = 0.2
) -> np.ndarray:
    """Create exponential decay distribution."""
    if not (0 <= last_item_prob <= 1):
        raise ValueError(f"last_item_prob={last_item_prob}")

    weights = np.exp(np.linspace(0, -temperature, size - 1))
    normalized_weights = weights / weights.sum() * (1 - last_item_prob)

    distribution = np.zeros(size)
    distribution[:-1] = normalized_weights
    distribution[-1] = last_item_prob

    return distribution


def calculate_ratio(ratio_string: str) -> float:
    """Calculate aspect ratio from string."""
    try:
        x, y = ratio_string.split(":")
        x = float(x)
        y = float(y)
        if y == 0:
            raise ValueError("Cannot divide by zero")
        return y / x
    except ValueError as e:
        raise ValueError(f"Invalid ratio format: {e}")


# =============================================================================
# Neural Painter
# =============================================================================

class NeuralPainter(BasePainter):
    """
    Neural painter using MLP-predicted strokes.
    
    Merged from NeuralSVG standalone:
    - model.py (SketchModel)
    - painter.py (Painter)
    - painter_nerf.py (PainterNerf)
    - nerf_mlp_multi.py (NerfMLPMulti)
    
    Features:
    - MLP predicts stroke positions and colors
    - Toggle color mode for training with different backgrounds
    - Toggle aspect ratio for different canvas ratios
    - Nested dropout for progressive rendering
    - Gaussian positional embeddings
    - Attention-based stroke initialization
    """

    def __init__(
        self,
        # Base parameters
        num_strokes: Optional[int] = None,
        canvas_size: Optional[int] = None,
        device: str = "cuda",
        num_segments: int = 4,
        control_points_per_seg: int = 2,
        width: float = 1.5,
        is_closed: bool = True,
        radius: float = 0.05,
        # Pretrain config
        sd_model: Optional[str] = None,
        lora_weights: Optional[str] = None,
        # MLP config
        mlp_dim: int = 128,
        mlp_num_layers: int = 2,
        input_dim: int = 128,
        use_nested_dropout: bool = True,
        truncation_start_idx: int = 4,
        points_prediction_scale: float = 0.1,
        nested_dropout_sampling_method: str = "uniform",
        # Color config
        use_color: bool = True,
        toggle_color: bool = False,
        toggle_color_method: str = "rgb",
        toggle_color_input_dim: int = 12,
        toggle_color_bg_colors: Optional[List[str]] = None,
        toggle_color_init_eps: float = 0.1,
        toggle_sample_random_color_prob: float = 0.0,
        # Aspect ratio config
        toggle_aspect_ratio: bool = False,
        toggle_aspect_ratio_values: Optional[List[str]] = None,
        aspect_ratio_emb_dim: Optional[int] = None,
        # Dropout config
        use_dropout_value: bool = False,
        dropout_emb_dim: int = 16,
        # Other
        use_background: bool = True,
        regular_polygon_closed_shape_init: bool = True,
        checkpoint_path: Optional[Path] = None,
    ):
        
        # Validate required parameters
        if num_strokes is None or canvas_size is None:
            raise ValueError("num_strokes and canvas_size are required")
        
        super().__init__(num_strokes, canvas_size, device)
        
        # Store configuration
        self.canvas_width = canvas_size
        self.canvas_height = canvas_size
        self.num_segments = num_segments
        self.control_points_per_seg = control_points_per_seg
        self.width = width
        self.is_closed = is_closed
        self.radius = radius
        self.use_background = use_background
        self.regular_polygon_closed_shape_init = regular_polygon_closed_shape_init
        
        # Pretrain config
        self.sd_model = sd_model
        self.lora_weights = lora_weights

        # MLP config
        self.mlp_dim = mlp_dim
        self.mlp_num_layers = mlp_num_layers
        self.input_dim = input_dim
        self.use_nested_dropout = use_nested_dropout
        self.truncation_start_idx = truncation_start_idx
        self.points_prediction_scale = points_prediction_scale
        self.nested_dropout_sampling_method = nested_dropout_sampling_method
        
        # Color config
        self.use_color = use_color
        self.toggle_color = toggle_color
        self.toggle_color_method = toggle_color_method
        self.toggle_color_input_dim = toggle_color_input_dim
        self.toggle_color_bg_colors = toggle_color_bg_colors or ["white"]
        self.toggle_color_init_eps = toggle_color_init_eps
        self.toggle_sample_random_color_prob = toggle_sample_random_color_prob
        
        # Aspect ratio config
        self.toggle_aspect_ratio = toggle_aspect_ratio
        self.toggle_aspect_ratio_values = toggle_aspect_ratio_values or ["1:1"]
        self.aspect_ratio_emb_dim = aspect_ratio_emb_dim
        
        # Dropout config
        self.use_dropout_value = use_dropout_value
        self.dropout_emb_dim = dropout_emb_dim
        
        # Compute total points per shape
        self.num_control_points_per_shape = (
            torch.zeros(num_segments, dtype=torch.long) + control_points_per_seg
        )
        self.shape_to_num_points = []
        for _ in range(num_strokes):
            num_points = self._get_total_num_points_static(
                control_points_per_seg, num_segments, is_closed
            )
            self.shape_to_num_points.append(num_points)
        self.total_num_points = max(self.shape_to_num_points)
        
        self.num_control_points = (
            torch.zeros(num_segments, dtype=torch.long) + control_points_per_seg
        )
        
        # Initialize radiuses
        self.radiuses = np.full(num_strokes, radius)
        
        # Color mapping
        self.color_name_to_value_map = self._get_color_name_to_value_map()
        
        # Initialize MLP
        self.mlp = self._initialize_mlp()
        if checkpoint_path is None:
            self.mlp.apply(init_weights)
        
        # Load checkpoint if provided
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)
        
        # Will be initialized in init_strokes or by attention map
        self.points_init = None
        self.init_widths = None
        self.strokes = []
        self.shape_groups = []
        self.shapes_init_colors = None
        self.stroke_idxs = None  # Normalized positions from attention map
        self.inds = None  # Pixel positions from attention map
        self.bg_inds = None  # Background pixel positions
        self.image = None  # Target image (PIL)
        self.image_tensor = None  # Target image tensor
        self.attn_map = None  # Attention map
        self.attn_map_soft_list = None
        self.background_attn_map_soft_list = None
        
        # Debug colors for visualization
        self.strokes_constant_colors = [
            torch.FloatTensor(np.random.uniform(size=[4]))
            for _ in range(num_strokes)
        ]

    def _initialize_mlp(self) -> NerfMLPMulti:
        """Initialize NeRF MLP model."""
        return NerfMLPMulti(
            num_strokes=self.num_strokes,
            total_num_points=self.total_num_points,
            intermediate_dim=self.mlp_dim,
            num_layers=self.mlp_num_layers,
            use_nested_dropout=self.use_nested_dropout,
            use_color=self.use_color,
            truncation_start_idx=self.truncation_start_idx,
            input_dim=self.input_dim,
            points_prediction_scale=self.points_prediction_scale,
            use_dropout_value=self.use_dropout_value,
            dropout_emb_dim=self.dropout_emb_dim,
            toggle_color=self.toggle_color,
            toggle_color_method=self.toggle_color_method,
            toggle_color_input_dim=self.toggle_color_input_dim,
            toggle_color_bg_colors=self.toggle_color_bg_colors,
            color_name_to_value_map=self.color_name_to_value_map,
            toggle_aspect_ratio=self.toggle_aspect_ratio,
            toggle_aspect_ratio_values=self.toggle_aspect_ratio_values,
            aspect_ratio_emb_dim=self.aspect_ratio_emb_dim,
            device=self.device,
        ).to(self.device)

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load model weights from checkpoint."""
        cp_path = Path(checkpoint_path)
        if not cp_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {cp_path}")

        print(f"Loading checkpoint from: {cp_path}")
        checkpoint = torch.load(cp_path, map_location=self.device)

        missing_keys, unexpected_keys = self.mlp.load_state_dict(
            checkpoint["state_dict"], strict=False
        )

        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")

        print("Checkpoint loaded successfully")

    @staticmethod
    def _get_total_num_points_static(
        control_points_per_seg: int, num_segments: int, is_closed: bool
    ) -> int:
        """Calculate total number of points per stroke."""
        num_points = control_points_per_seg + 2  # First segment
        num_points += (control_points_per_seg + 1) * (num_segments - 1)
        num_points -= int(is_closed)  # Closed shapes share endpoint
        return num_points

    def _get_color_name_to_value_map(self) -> Dict[str, torch.Tensor]:
        """Create mapping from color names to normalized RGB values."""
        color_map = {}
        
        # Custom colors
        custom_colors = {
            "light red": [238, 75, 43],
            "light green": [144, 238, 144],
            "light blue": [167, 199, 231],
            "red pastel": [238, 75, 43],
            "green pastel": [144, 238, 144],
            "blue pastel": [167, 199, 231],
            "sky blue": [135, 206, 235],
            "turquoise": [64, 224, 208],
            "brown": [165, 42, 42],
            "light gray": [144, 238, 144],
            "gold": [255, 215, 0],
            "turquoise green": [69, 196, 176],
            "tea green": [218, 253, 186],
            "charcoal blue": [37, 54, 89],
            "light salmon": [242, 116, 87],
            "dark magenta": [115, 18, 81],
            "lavender blue": [160, 163, 217],
            "light sky blue": [167, 208, 217],
        }
        
        for name, rgb in custom_colors.items():
            color_map[name] = torch.tensor(rgb + [255], dtype=torch.float32) / 255
        
        # Web colors
        web_colors = [
            "aqua", "black", "blue", "fuchsia", "green", "gray", "lime",
                      "maroon", "navy", "olive", "purple", "red", "silver", "teal",
            "white", "yellow",
        ]
        
        for color_name in web_colors:
            try:
                rgb = webcolors.name_to_rgb(color_name)
                color_map[color_name] = torch.tensor(
                    [rgb.red, rgb.green, rgb.blue, 255], dtype=torch.float32
                ) / 255
            except ValueError:
                pass
        
        return color_map

    # =========================================================================
    # Initialization methods (from Painter)
    # =========================================================================

    def init_strokes(
        self,
        phases: Optional[List[Dict]] = None,
        attention_inds: Optional[List] = None,
        target_image: Optional[torch.Tensor] = None,
        pixel_inds: Optional[List[Tuple[int, int]]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize strokes for neural painter.
        
        This method can be called:
        1. With attention_inds/pixel_inds for attention-based initialization
        2. Without (will use random initialization or from image if available)
        
        Args:
            phases: Phase configurations (for BasePainter compatibility)
            attention_inds: Normalized positions from attention map - list of (x, y) in [0, 1]
            target_image: Target image tensor (1, C, H, W) for color initialization
            pixel_inds: Pixel coordinates (h, w) from attention map for color sampling
        """
        # Initialize phase tracking if phases provided
        if phases:
            self._init_phase_tracking(phases)
        
        # Store attention indices and image
        if attention_inds is not None:
            self.stroke_idxs = attention_inds
        if pixel_inds is not None:
            self.inds = pixel_inds
        if target_image is not None:
            self.image_tensor = target_image
        
        # Initialize stroke colors
        if self.toggle_color:
            self.shapes_init_colors = self._init_colors_toggled(eps=self.toggle_color_init_eps)
        else:
            self.shapes_init_colors = self._init_colors()
        
        # Initialize strokes
        self.strokes, self.shape_groups, points_init, self.shape_groups_colored = self._init_strokes_internal()
        self.points_init = torch.stack(points_init)
        self.init_widths = torch.ones((1, self.num_strokes)).to(self.device) * self.width
        
        self.strokes_initialized = True

    def _get_phase_num_strokes(self, phase: 'PhaseConfig', default: int) -> int:
        """Helper function to get num_strokes from phase (PhaseConfig)."""
        return phase.num_strokes if phase.num_strokes is not None else default
    
    def _init_phase_tracking(self, phases: List['PhaseConfig']) -> None:
        """Initialize phase tracking from phase configurations."""
        self.stroke_phase_index = []
        self.phase_prefix_counts = []
        
        prev_count = 0
        for phase_idx, phase in enumerate(phases):
            phase_strokes = self._get_phase_num_strokes(phase, self.num_strokes)
            for i in range(prev_count, phase_strokes):
                self.stroke_phase_index.append(phase_idx)
            self.phase_prefix_counts.append(phase_strokes)
            prev_count = phase_strokes

    def _init_strokes_internal(self) -> Tuple[List, List, List, List]:
        """Initialize strokes, shape_groups, points_init."""
        strokes, shape_groups, points_init, shape_groups_colored = [], [], [], []
        
        for idx in range(self.num_strokes):
            # Get initial color
            if self.toggle_color:
                stroke_color_init = self.shapes_init_colors[self.toggle_color_bg_colors[0]][idx]
            else:
                stroke_color_init = self.shapes_init_colors[idx]
            
            stroke_color_black = torch.tensor([0.0, 0.0, 0.0, 1.0])
            stroke_color_colored = self.strokes_constant_colors[idx]
            
            # Get path or shape
            path, points = self._get_shape(idx) if self.is_closed else self._get_path(idx)
            strokes.append(path)
            points_init.append(points)
            
            shape_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(strokes) - 1]),
                fill_color=stroke_color_init if self.is_closed else None,
                stroke_color=stroke_color_black if not self.is_closed else None,
            )
            shape_groups.append(shape_group)
            
            shape_group_colored = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(strokes) - 1]),
                fill_color=stroke_color_colored if self.is_closed else None,
                stroke_color=stroke_color_colored,
            )
            shape_groups_colored.append(shape_group_colored)
        
        return strokes, shape_groups, points_init, shape_groups_colored

    def _get_path(self, idx: int) -> Tuple[pydiffvg.Path, torch.Tensor]:
        """Generate an open path stroke."""
        points = []
        total_num_points = self.shape_to_num_points[idx]
        num_control_points = self.num_control_points_per_shape
        
        # Get initial position
        p0 = (
            self.stroke_idxs[idx]
            if self.stroke_idxs and idx < len(self.stroke_idxs)
            else (random.random(), random.random())
        )
        if isinstance(p0, torch.Tensor):
            p0 = tuple(p0.tolist())
        elif isinstance(p0, (list, np.ndarray)):
            p0 = tuple(p0)
        
        points.append(p0)
        radius = self.radiuses[idx] if idx < len(self.radiuses) else self.radius
        
        for _ in range(total_num_points - 1):
            p1 = (
                p0[0] + radius * (random.random() - 0.5),
                p0[1] + radius * (random.random() - 0.5),
            )
            points.append(p1)
            p0 = p1
        
        points = torch.tensor(points, device=self.device, dtype=torch.float32)
        points[:, 0] *= self.canvas_width
        points[:, 1] *= self.canvas_height
        
        path = pydiffvg.Path(
            num_control_points=num_control_points,
            points=points,
            stroke_width=torch.tensor(self.width).to(self.device),
            is_closed=self.is_closed,
        )
        return path, points

    def _get_shape(self, idx: int) -> Tuple[pydiffvg.Path, torch.Tensor]:
        """Generate a closed shape stroke."""
        total_num_points = self.shape_to_num_points[idx]
        num_control_points = self.num_control_points_per_shape
        
        # Get initial position
        p0 = (
            self.stroke_idxs[idx]
            if self.stroke_idxs and idx < len(self.stroke_idxs)
            else (random.random(), random.random())
        )
        if isinstance(p0, torch.Tensor):
            p0 = tuple(p0.tolist())
        elif isinstance(p0, (list, np.ndarray)):
            p0 = tuple(p0)
        
        radius = self.radiuses[idx] if idx < len(self.radiuses) else self.radius
        
        if self.regular_polygon_closed_shape_init:
            # Regular polygon
            points = self.create_polygon(center=p0, radius=radius, num_edges=self.num_segments * 3)
        else:
            # Random closed shape
            points = [p0]
            current_p = p0
            for _ in range(total_num_points - 1):
                p1 = (
                    current_p[0] + radius * (random.random() - 0.5),
                    current_p[1] + radius * (random.random() - 0.5),
                )
                points.append(p1)
                current_p = p1
        
        # Convert to tensor if needed (create_polygon returns list)
        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, device=self.device, dtype=torch.float32)
        
        points[:, 0] *= self.canvas_width
        points[:, 1] *= self.canvas_height
        
        path = pydiffvg.Path(
            num_control_points=num_control_points,
            points=points,
            stroke_width=torch.tensor(self.width).to(self.device),
            is_closed=self.is_closed,
        )
        return path, points

    @staticmethod
    def create_polygon(center: Tuple[float, float], radius: float, num_edges: int) -> List[Tuple[float, float]]:
        """Create regular polygon vertices."""
        cx, cy = center
        angle = 2 * np.pi / num_edges
        vertices = [
            (cx + radius * np.cos(i * angle), cy + radius * np.sin(i * angle))
            for i in range(num_edges)
        ]
        return vertices

    def _init_colors(self) -> torch.Tensor:
        """Initialize stroke colors from target image."""
        init_color_lst = []
        for p in range(self.num_strokes):
            if self.inds is not None and self.image_tensor is not None and p < len(self.inds):
                href, wref = self.inds[p]
                init_color = torch.clone(self.image_tensor[0, :, href, wref])
            else:
                init_color = torch.tensor([random.uniform(0, 1) for _ in range(3)])
            init_opacity = torch.tensor([1.0])
            init_color_lst.append(torch.cat((init_color, init_opacity)))
        
        init_color_tns = torch.stack(init_color_lst).to(self.device)
        return init_color_tns

    def _init_colors_toggled(self, eps: float = 0.1) -> Dict[str, torch.Tensor]:
        """Initialize stroke colors with toggle color support."""
        ans = {}
        
        for color_name in self.toggle_color_bg_colors:
            init_color_lst = []
            for p in range(self.num_strokes):
                if self.inds is not None and self.image_tensor is not None and p < len(self.inds):
                    href, wref = self.inds[p]
                    init_color = torch.clone(self.image_tensor[0, :, href, wref])
                else:
                    init_color = torch.tensor([random.uniform(0, 1) for _ in range(3)])
                init_opacity = torch.tensor([1.0])
                init_color_lst.append(torch.cat((init_color, init_opacity)))
        
            init_color_tns = torch.stack(init_color_lst)
            init_color_tns = init_color_tns + torch.normal(mean=0.0, std=eps, size=init_color_tns.shape)
            init_color_tns = torch.clamp(init_color_tns, min=0, max=1)
            init_color_tns = init_color_tns.to(self.device)
            
            ans[color_name] = init_color_tns
        
        return ans

    # =========================================================================
    # Core rendering methods (from PainterNerf)
    # =========================================================================

    def mlp_pass(
        self,
        mode: str = "train",
        eps: float = 1e-4,
        truncation_indices: Optional[List[int]] = None,
        sub_layers_sizes: Optional[List[int]] = None,
        toggle_color_value: Optional[ColorValue] = None,
        render_without_background: bool = False,
        toggle_aspect_ratio_value: Optional[str] = None,
        remove_fill_color: bool = False,
        indices_to_pass: Optional[List[int]] = None,
        debug_aspect_ratio: bool = False,
        aspect_ratio_white_rects: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, None, torch.Tensor, torch.Tensor]:
        """
        Generate vector graphics through MLP forward pass.
            
        Returns:
            Tuple of (img, all_points, None, opacities, stroke_colors)
        """
        # Initialize toggle values
        if toggle_color_value is None and self.toggle_color:
                toggle_color_value = random.choice(self.toggle_color_bg_colors)
        
        if toggle_aspect_ratio_value is None and self.toggle_aspect_ratio:
            toggle_aspect_ratio_value = "1:1"
        
        # Validate sub-layers
        if sub_layers_sizes is not None:
            if not isinstance(sub_layers_sizes, list):
                raise ValueError(f"sub_layers_sizes must be a list")
            if sum(sub_layers_sizes) != self.num_strokes:
                raise ValueError(f"sub_layers_sizes must sum to {self.num_strokes}")
        
        # Handle nested dropout
        truncation_idx = None
        if truncation_indices is not None:
            truncation_idx = truncation_indices[0]
        elif self.use_nested_dropout:
            truncation_idx = NerfMLPMulti.sample_truncation_idx(
                self.num_strokes,
                start_idx=self.truncation_start_idx,
                sampling_method=self.nested_dropout_sampling_method,
            )
        
        # Get MLP predictions
        aspect_ratio_value = "1:1" if debug_aspect_ratio else toggle_aspect_ratio_value
        points, widths, opacities, colors = self.mlp(
            truncation_idx=truncation_idx,
            toggle_color_value=toggle_color_value,
            toggle_aspect_ratio_value=aspect_ratio_value,
            indices_to_pass=indices_to_pass,
        )
        
        # Process points
        all_points = 0.5 * (points + 1.0) * self.canvas_width
        all_points = all_points + eps * torch.randn_like(all_points)
        
        # Apply aspect ratio
        if toggle_aspect_ratio_value == "4:1":
            ratio = 0.25
            all_points[:, :, :, 1] = all_points[:, :, :, 1] * ratio + (1 - ratio) * (self.canvas_height / 2)
        
        # Process widths and opacities
        widths = self.mlp.apply_nested_dropout(self.init_widths.clone().detach(), truncation_idx=truncation_idx)
        opacities = self.mlp.apply_nested_dropout(opacities.clone().detach(), truncation_idx=truncation_idx)
        
        # Create shapes and shape groups
        shapes, shape_groups, stroke_colors = self._create_shapes_and_groups(
            all_points, widths[0], colors, opacities,
            truncation_idx, indices_to_pass, mode,
            remove_fill_color, toggle_color_value,
            render_without_background, toggle_aspect_ratio_value,
            aspect_ratio_white_rects,
        )
        
        # Render
        img = self.render_scene(shapes, shape_groups)
        self.strokes = shapes.copy()
        self.shape_groups = shape_groups.copy()
        
        return img, all_points, None, opacities, stroke_colors

    def _create_shapes_and_groups(
        self,
        all_points: torch.Tensor,
        widths: torch.Tensor,
        colors: Optional[torch.Tensor],
        opacities: torch.Tensor,
        truncation_idx: Optional[int],
        indices_to_pass: Optional[List[int]],
        mode: str,
        remove_fill_color: bool,
        toggle_color_value: Optional[ColorValue],
        render_without_background: bool,
        toggle_aspect_ratio_value: Optional[str],
        aspect_ratio_white_rects: bool,
    ) -> Tuple[ShapeList, ShapeGroupList, torch.Tensor]:
        """Create shapes and shape groups for rendering."""
        shapes: ShapeList = []
        shape_groups: ShapeGroupList = []
        stroke_color_lst = []
        
        # Add background if needed
        if self.use_background:
            background_color = self._get_background_color(toggle_color_value, render_without_background)
            shapes.append(self._create_background_shape())
            shape_groups.append(pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes) - 1]),
                fill_color=background_color,
            ))
        
        # Get number of shapes
        if indices_to_pass is not None:
            number_of_shapes = len(indices_to_pass)
        elif truncation_idx is not None:
            number_of_shapes = truncation_idx
        else:
            number_of_shapes = self.num_strokes
        
        for p in range(number_of_shapes):
            actual_stroke_idx = indices_to_pass[p] if indices_to_pass is not None else p
            
            width = (
                widths[actual_stroke_idx].clone().detach()
                if mode != "init"
                else torch.tensor(self.width)
            )
            
            path = pydiffvg.Path(
                num_control_points=self.num_control_points_per_shape,
                points=all_points[:, p, :].reshape((-1, 2)),
                stroke_width=width,
                is_closed=self.is_closed,
            )
            
            # Color indices
            color_idx = p if indices_to_pass is not None else actual_stroke_idx
            opacity_idx = actual_stroke_idx
            stroke_color = self._create_stroke_color(colors, opacities, color_idx, opacity_idx)
            stroke_color_lst.append(stroke_color)
            
            stroke_color_to_render = self._get_stroke_color_to_render(stroke_color, remove_fill_color)
            
            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes) - 1]),
                fill_color=stroke_color if (self.is_closed and not remove_fill_color) else None,
                stroke_color=stroke_color_to_render,
            )
            shape_groups.append(path_group)
        
        # Add aspect ratio rectangles if needed
        if self.toggle_aspect_ratio and toggle_aspect_ratio_value == "4:1":
            self._add_aspect_ratio_rects(shapes, shape_groups, toggle_color_value, aspect_ratio_white_rects)
        
        return shapes, shape_groups, torch.stack(stroke_color_lst)

    def _get_background_color(self, toggle_color_value: Optional[ColorValue], render_without_background: bool) -> torch.Tensor:
        """Get background color based on settings."""
        if render_without_background:
            return self.color_name_to_value_map["white"]
        elif toggle_color_value is None:
            return self.mlp.background_color_params
        elif isinstance(toggle_color_value, str):
            return self.color_name_to_value_map[toggle_color_value]
        elif isinstance(toggle_color_value, torch.Tensor):
            return toggle_color_value
        else:
            raise ValueError(f"Invalid toggle_color_value type: {type(toggle_color_value)}")

    def _create_background_shape(self) -> pydiffvg.Rect:
        """Create background rectangle shape."""
        return pydiffvg.Rect(
            p_min=torch.tensor([0.0, 0.0]),
            p_max=torch.tensor([float(self.canvas_width), float(self.canvas_height)]),
        )

    def _create_stroke_color(
        self, colors: Optional[torch.Tensor], opacities: torch.Tensor, color_idx: int, opacity_idx: int
    ) -> torch.Tensor:
        """Create stroke color with opacity."""
        stroke_color_rgb = (
            colors[:, color_idx]
            if colors is not None
            else torch.tensor([0, 0, 0], device=self.device)
        )
        stroke_color_alpha = (
            opacities[:, opacity_idx]
            if opacities is not None
            else torch.tensor([1], device=self.device)
        )
        return torch.cat((stroke_color_rgb, stroke_color_alpha))

    def _get_stroke_color_to_render(self, stroke_color: torch.Tensor, remove_fill_color: bool) -> Optional[torch.Tensor]:
        """Get stroke color for rendering."""
        if self.is_closed:
            return None
        if remove_fill_color:
            return torch.tensor([0, 0, 0, 1], device=self.device)
        return stroke_color

    def _add_aspect_ratio_rects(
        self,
        shapes: ShapeList,
        shape_groups: ShapeGroupList,
        toggle_color_value: Optional[ColorValue],
        aspect_ratio_white_rects: bool,
    ) -> None:
        """Add aspect ratio rectangles."""
        rects_fill_color = (
            self._get_background_color(toggle_color_value, False)
            if not aspect_ratio_white_rects
            else self.color_name_to_value_map["white"]
        )
        
        # Top rectangle
        shapes.append(pydiffvg.Rect(
            p_min=torch.tensor([0.0, 0.0]),
            p_max=torch.tensor([self.canvas_width, (self.canvas_height * 3) / 8]),
        ))
        shape_groups.append(pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([len(shapes) - 1]),
            fill_color=rects_fill_color,
        ))
        
        # Bottom rectangle
        shapes.append(pydiffvg.Rect(
            p_min=torch.tensor([0, (self.canvas_height * 5) / 8]),
            p_max=torch.tensor([self.canvas_width, self.canvas_height]),
        ))
        shape_groups.append(pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([len(shapes) - 1]),
            fill_color=rects_fill_color,
        ))

    def render_scene(self, shapes: List, shape_groups: List) -> torch.Tensor:
        """Render shapes to image."""
        _render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            self.canvas_width, self.canvas_height, shapes, shape_groups
        )
        img = _render(
            self.canvas_width,
            self.canvas_height,
            2, 2,  # num_samples
            0,  # seed
            None,
            *scene_args,
        )
        return img

    def handle_raw_image(self, img: torch.Tensor) -> torch.Tensor:
        """Process raw rendered image to NCHW format."""
        opacity = img[:, :, 3:4]
        img = opacity * img[:, :, :3] + torch.ones(
            img.shape[0], img.shape[1], 3, device=self.device
        ) * (1 - opacity)
        img = img[:, :, :3].unsqueeze(0)
        img = img.permute(0, 3, 1, 2).to(self.device)  # NHWC -> NCHW
        return img
    
    def _random_toggle_color(self) -> Optional[ColorValue]:
        """Randomly select a toggle color value."""
        if self.toggle_color and self.toggle_color_bg_colors:
            return random.choice(self.toggle_color_bg_colors)
        return None

    # =========================================================================
    # High-level API (matching NeuralSVG interface)
    # =========================================================================

    def get_image(
        self,
        mode: str = "train",
        num_strokes: Optional[int] = None,
        truncation_indices: Optional[List[int]] = None,
        sub_layers_sizes: Optional[List[int]] = None,
        toggle_color_value: Optional[ColorValue] = None,
        render_without_background: bool = False,
        toggle_aspect_ratio_value: Optional[str] = None,
        remove_fill_color: bool = False,
        aspect_ratio_white_rects: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, None, torch.Tensor, torch.Tensor]:
        """
        Get rendered image.
        
        Returns:
            Tuple of (img, points, None, opacities, shape_color_rgb)
        """
        if self.mlp is None:
            raise ValueError("MLP not initialized")
        truncation_indices = [num_strokes] if truncation_indices is None else truncation_indices
        if mode != "init":
            img, points, _, opacities, shape_color_rgb = self.mlp_pass(
                mode,
                truncation_indices=truncation_indices,
                sub_layers_sizes=sub_layers_sizes,
                toggle_color_value=toggle_color_value,
                render_without_background=render_without_background,
                toggle_aspect_ratio_value=toggle_aspect_ratio_value,
                remove_fill_color=remove_fill_color,
                aspect_ratio_white_rects=aspect_ratio_white_rects,
            )
        else:
            raise Exception("'init' mode deprecated")
        
        img = self.handle_raw_image(img)
        return img
        # return img, points, None, opacities, shape_color_rgb

    def get_points(
        self,
        mode: str = "train",
        num_strokes: Optional[int] = None,
        truncation_indices: Optional[List[int]] = None,
        toggle_color_value: Optional[ColorValue] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predicted points and colors from MLP.
            
        Returns:
            Tuple of (points, shape_color_rgb)
        """
        truncation_indices = [num_strokes] if truncation_indices is None else truncation_indices
        _, points, _, _, shape_color_rgb = self.mlp_pass(
            mode,
            truncation_indices=truncation_indices,
            toggle_color_value=toggle_color_value,
        )
        return points, shape_color_rgb

    def render_warp(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render with warped points."""
        img = self.render_scene(self.strokes, self.shape_groups)
        img_colored = self.render_scene(self.strokes, self.shape_groups_colored)
        all_points = torch.stack([s.points for s in self.strokes]).unsqueeze(0)
        return img, all_points, img_colored

    # =========================================================================
    # Save methods
    # =========================================================================

    def save_svg(
        self,
        path: Union[str, Path],
        num_strokes: Optional[int] = None,
        save_groups: bool = False,
        phases: Optional[List[Dict]] = None,
        toggle_color_value: Optional[ColorValue] = None,
    ) -> List[str]:
        """
        Save strokes to SVG file(s).
        
        Args:
            path: Output path (without extension for multiple files).
            num_strokes: Number of strokes to save.
            save_groups: Whether to save phase-colored version.
            phases: Phase configurations for coloring.
            toggle_color_value: Optional toggle color value for rendering.
            
        Returns:
            List of saved file paths.
        """
        import matplotlib.cm as cm
        
        path = str(path)
        if path.endswith('.svg'):
            base_path = path[:-4]
        else:
            base_path = path
        
        # Ensure parent directory exists
        parent_dir = Path(base_path).parent
        parent_dir.mkdir(parents=True, exist_ok=True)
        
        # Render to update strokes with correct num_strokes
        truncation_indices = [num_strokes] if num_strokes is not None else None
        _ = self.get_image(
            mode="train",
            truncation_indices=truncation_indices,
            toggle_color_value=toggle_color_value,
            # render_without_background=True,
        )
        
        # Get shapes and shape_groups to save
        shapes = self.strokes
        shape_groups = self.shape_groups
        
        # Store original colors
        original_colors = []
        for sg in shape_groups:
            if sg.fill_color is not None:
                original_colors.append(('fill', sg.fill_color.detach().clone()))
            elif sg.stroke_color is not None:
                original_colors.append(('stroke', sg.stroke_color.detach().clone()))
            else:
                original_colors.append((None, None))
        
        svg_files = []
        
        if save_groups and phases and len(phases) > 1:
            # Save with phase colors
            cmap = cm.get_cmap('tab10', 10)
            for i, sg in enumerate(shape_groups):
                # Skip background rectangle - keep it white
                if i < len(shapes) and isinstance(shapes[i], pydiffvg.Rect) and i == 0:
                    continue
                
                if self.stroke_phase_index is not None and i < len(self.stroke_phase_index):
                    phase_idx = self.stroke_phase_index[i]
                else:
                    # Determine phase from stroke index and phases
                    phase_idx = 0
                    for p_idx, phase in enumerate(phases):
                        phase_strokes = self._get_phase_num_strokes(phase, self.num_strokes)
                        if i < phase_strokes:
                            phase_idx = p_idx
                            break
                
                phase_idx = min(phase_idx, len(phases) - 1)
                rgba = cmap.colors[phase_idx]
                color_tensor = torch.tensor(
                    [float(rgba[0]), float(rgba[1]), float(rgba[2]), 1.0],
                    device=self.device
                )
                
                if self.is_closed:
                    sg.fill_color = color_tensor
                else:
                    sg.stroke_color = color_tensor
            
            group_path = f"{base_path}_group.svg"
            pydiffvg.save_svg(
                group_path,
                self.canvas_width, self.canvas_height,
                shapes, shape_groups
            )
            svg_files.append(group_path)
            
            # Restore original colors after saving group version
            for sg, (color_type, orig_color) in zip(shape_groups, original_colors):
                if orig_color is not None:
                    if color_type == 'fill':
                        sg.fill_color = orig_color
                    elif color_type == 'stroke':
                        sg.stroke_color = orig_color
        
        # Save with original colors
        svg_path = f"{base_path}.svg"
        pydiffvg.save_svg(
            svg_path,
            self.canvas_width,
            self.canvas_height,
            shapes, 
            shape_groups
        )
        svg_files.append(svg_path)
        
        return svg_files
    
    def save_svg_single(
        self,
        output_dir: Path,
        name: str,
        truncation_indices: Optional[List[int]] = None,
        toggle_color_value: Optional[ColorValue] = None,
        show_log: bool = True,
    ) -> None:
        """Save SVG file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if truncation_indices is not None or toggle_color_value is not None:
            _ = self.get_image(
                mode="train",
                truncation_indices=truncation_indices,
            toggle_color_value=toggle_color_value,
        )
        
        if show_log:
            print(f"Save SVG @ {output_dir}/{name}.svg")
        pydiffvg.save_svg(
            f"{output_dir}/{name}.svg",
            self.canvas_width,
            self.canvas_height,
            self.strokes,
            self.shape_groups,
        )

    def save_png(
        self,
        output_dir: Path,
        name: str,
        truncation_indices: Optional[List[int]] = None,
        toggle_color_value: Optional[ColorValue] = None,
        background_color: Optional[str] = "white",
        show_log: bool = True,
    ) -> None:
        """Save PNG file."""
        from PIL import Image
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        render_without_background = (background_color is None)
        
        self.mlp.eval()
        with torch.no_grad():
            img_raw, _, _, _, _ = self.mlp_pass(
                mode="eval",
                truncation_indices=truncation_indices,
                toggle_color_value=toggle_color_value,
                render_without_background=render_without_background,
            )
        self.mlp.train()
        
        img_np = img_raw.cpu().detach().numpy() if isinstance(img_raw, torch.Tensor) else img_raw
        img_np = np.clip(img_np, 0, 1)
        img_np = (img_np * 255).astype(np.uint8)
        
        if background_color is not None and background_color != "transparent":
            if background_color == "white":
                bg_rgb = np.array([255, 255, 255], dtype=np.uint8)
            elif background_color == "black":
                bg_rgb = np.array([0, 0, 0], dtype=np.uint8)
            else:
                try:
                    bg_rgb_tuple = webcolors.name_to_rgb(background_color)
                    bg_rgb = np.array([bg_rgb_tuple.red, bg_rgb_tuple.green, bg_rgb_tuple.blue], dtype=np.uint8)
                except (ValueError, AttributeError):
                    bg_rgb = np.array([255, 255, 255], dtype=np.uint8)
            
            alpha = img_np[:, :, 3:4] / 255.0
            rgb = img_np[:, :, :3]
            img_np = (alpha * rgb + (1 - alpha) * bg_rgb).astype(np.uint8)
            img_np = np.concatenate([img_np, np.full((img_np.shape[0], img_np.shape[1], 1), 255, dtype=np.uint8)], axis=2)
        
        img_pil = Image.fromarray(img_np, mode='RGBA')
        output_path = output_dir / f"{name}.png"
        img_pil.save(output_path)
        
        if show_log:
            print(f"Save PNG @ {output_path}")

    # =========================================================================
    # BasePainter interface methods
    # =========================================================================

    def render(self, num_strokes: Optional[int] = None, return_alpha: bool = False) -> torch.Tensor:
        """Render strokes to image (BasePainter interface)."""
        truncation_indices = [num_strokes] if num_strokes is not None else None
        img, _, _, _, _ = self.get_image(truncation_indices=truncation_indices)
        return img

    def get_points_parameters(self) -> List[torch.nn.Parameter]:
        """Get MLP parameters."""
        if self.mlp is None:
            return []
        return list(self.mlp.parameters())

    def get_parameters(self) -> List[torch.nn.Parameter]:
        """Get all parameters (alias for get_points_parameters)."""
        return self.get_points_parameters()

    def get_color_parameters(self) -> List[torch.nn.Parameter]:
        """Get color parameters (handled by MLP)."""
        return []

    def get_width_parameters(self) -> List[torch.nn.Parameter]:
        """Get width parameters (not used for neural painter)."""
        return []

    def get_phase_stroke_count(self, phase_idx: int, phases: List['PhaseConfig']) -> int:
        """Get cumulative stroke count for a phase."""        
        if hasattr(self, 'phase_prefix_counts') and self.phase_prefix_counts:
            return self.phase_prefix_counts[min(phase_idx, len(self.phase_prefix_counts) - 1)]
        
        phase = phases[phase_idx]
        return phase.num_strokes if phase.num_strokes is not None else self.num_strokes

    def overlay_loss(
        self,
        phases: List[Dict],
        loss_type: str = "dot",
        blur_sigma: float = 2.0,
        blur_kernel_size: int = 15,
        hinge_threshold: float = 0.5,
        save_debug_images: bool = False,
        debug_output_dir: Optional[Path] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute overlay loss between phase 1 strokes and delta strokes.
        
        This implementation uses mlp_pass with indices_to_pass to render
        specific strokes, following the coach.py approach.
        
        Args:
            phases: List of phase configurations.
            loss_type: Type of overlay loss ("dot", "dice").
            blur_sigma: Sigma for Gaussian blur (not used in this implementation).
            blur_kernel_size: Kernel size for Gaussian blur (not used).
            hinge_threshold: Threshold for hinge loss (not used).
            save_debug_images: Whether to save debug images.
            debug_output_dir: Directory for debug images.
            **kwargs: Additional arguments (ignored).
            
        Returns:
            Overlay loss value.
        """        
        assert len(phases) >= 2, "Overlay loss requires at least 2 phases"
        if len(phases) != 2:
            warnings.warn("We only support 2 phases for overlay loss, returning 0 loss")
            return torch.tensor(0.0, device=self.device)
        
        # Get stroke counts for phase A (first phase) and phase B (last phase)
        num_strokes_A = self._get_phase_num_strokes(phases[0], self.num_strokes // 2)
        num_strokes_B = self._get_phase_num_strokes(phases[-1], self.num_strokes)
        
        # Delta strokes are from num_strokes_A to num_strokes_B
        if num_strokes_A >= num_strokes_B:
            warnings.warn("num_strokes_A >= num_strokes_B, returning 0 loss")
            return torch.tensor(0.0, device=self.device)
        
        # Render phase 1 strokes (first num_strokes_A strokes)
        img_p1_raw, _, _, _, _ = self.mlp_pass(
            mode="train",
            truncation_indices=[num_strokes_A],
            toggle_color_value="white", # use white background
            render_without_background=True,
        )
        
        # Extract alpha channel as mask and exclude white background
        # img_p1_raw is (H, W, 4) from render_scene
        mask_p1 = img_p1_raw[:, :, 3]  # (H, W)
        rgb_p1 = img_p1_raw[:, :, :3]  # (H, W, 3)
        is_white_bg = (rgb_p1[:, :, 0] > 0.99) & (rgb_p1[:, :, 1] > 0.99) & (rgb_p1[:, :, 2] > 0.99)
        mask_p1 = mask_p1 * (~is_white_bg).float()
        
        # Render delta strokes (from num_strokes_A to num_strokes_B)
        delta_indices = list(range(num_strokes_A, num_strokes_B))
        img_delta_raw, _, _, _, _ = self.mlp_pass(
            mode="train",
            truncation_indices=[num_strokes_B],
            toggle_color_value="white", # use white background
            indices_to_pass=delta_indices,
            render_without_background=True,
        )
        
        # Extract alpha channel as mask and exclude white background
        mask_delta = img_delta_raw[:, :, 3]  # (H, W)
        rgb_delta = img_delta_raw[:, :, :3]  # (H, W, 3)
        is_white_bg_delta = (rgb_delta[:, :, 0] > 0.99) & (rgb_delta[:, :, 1] > 0.99) & (rgb_delta[:, :, 2] > 0.99)
        mask_delta = mask_delta * (~is_white_bg_delta).float()
        
        # Save debug images if requested
        if save_debug_images and debug_output_dir is not None:
            debug_output_dir = Path(debug_output_dir)
            debug_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert tensors to images and save
            img_p1_np = img_p1_raw.cpu().detach().numpy() if isinstance(img_p1_raw, torch.Tensor) else img_p1_raw
            img_delta_np = img_delta_raw.cpu().detach().numpy() if isinstance(img_delta_raw, torch.Tensor) else img_delta_raw
            
            img_p1_np = np.clip(img_p1_np, 0, 1)
            img_delta_np = np.clip(img_delta_np, 0, 1)
            img_p1_np = (img_p1_np * 255).astype(np.uint8)
            img_delta_np = (img_delta_np * 255).astype(np.uint8)
            
            # Save images (RGBA)
            img_p1_pil = Image.fromarray(img_p1_np, mode='RGBA')
            img_delta_pil = Image.fromarray(img_delta_np, mode='RGBA')
            img_p1_pil.save(debug_output_dir / "img_p1_raw.png")
            img_delta_pil.save(debug_output_dir / "img_delta_raw.png")
            
            # Save masks (grayscale)
            mask_p1_np = (mask_p1.cpu().detach().numpy() * 255).astype(np.uint8)
            mask_delta_np = (mask_delta.cpu().detach().numpy() * 255).astype(np.uint8)
            mask_p1_pil = Image.fromarray(mask_p1_np, mode='L')
            mask_delta_pil = Image.fromarray(mask_delta_np, mode='L')
            mask_p1_pil.save(debug_output_dir / "mask_p1.png")
            mask_delta_pil.save(debug_output_dir / "mask_delta.png")
            
            # Save intersection mask
            mask_dot = mask_p1 * mask_delta
            mask_dot_np = (mask_dot.cpu().detach().numpy() * 255).astype(np.uint8)
            mask_dot_pil = Image.fromarray(mask_dot_np, mode='L')
            mask_dot_pil.save(debug_output_dir / "mask_dot.png")
        
        # Compute intersection mask
        mask_dot = mask_p1 * mask_delta  # (H, W)
        
        # Compute loss based on loss_type
        if loss_type == "dot":
            # Dot product: L_dot = E[M_1 * M_2] = mean(M_1 * M_2)
            loss = mask_dot.mean()
        elif loss_type == "dice":
            # Soft Dice: Dice = (2 * Σ M₁ M₂) / (Σ M₁ + Σ M₂)
            sum_p1 = mask_p1.sum()
            sum_delta = mask_delta.sum()
            mask_dot_sum = mask_dot.sum()
            loss = (2 * mask_dot_sum) / (sum_p1 + sum_delta + 1e-8)
        elif loss_type == "iou":
            # Soft IoU loss
            intersection = mask_dot.sum()
            union = mask_p1.sum() + mask_delta.sum() - intersection
            loss = intersection / (union + 1e-8)
        elif loss_type == "hinge":
            # Hinge loss
            import torch.nn.functional as F
            loss = F.relu(mask_dot - hinge_threshold).mean()
        else:
            loss = torch.tensor(0.0, device=self.device)
        
        return loss

    def clamp_parameters(self) -> None:
        """No clamping needed for neural painter."""
        pass

    def _render_mask(self, start_idx: int, end_idx: int) -> torch.Tensor:
        """Render a binary mask for strokes."""
        if end_idx <= start_idx:
            return torch.zeros(self.canvas_size, self.canvas_size, device=self.device)
        
        img, _, _, _, _ = self.get_image(truncation_indices=[end_idx], render_without_background=True)
        img_gray = img[0].mean(dim=0)
        mask = 1.0 - img_gray
        return mask

    # =========================================================================
    # State dict methods
    # =========================================================================

    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for checkpointing."""
        state = super().state_dict()
        if self.mlp is not None:
            state['mlp_state'] = self.mlp.state_dict()
        if self.points_init is not None:
            state['points_init'] = self.points_init.cpu()
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from dictionary."""
        super().load_state_dict(state_dict)
        if 'mlp_state' in state_dict and self.mlp is not None:
            self.mlp.load_state_dict(state_dict['mlp_state'])
        if 'points_init' in state_dict:
            self.points_init = state_dict['points_init'].to(self.device)

    def pretrain(
        self,
        phases: List[PhaseConfig],
        data_cfg: Any,
        optim_cfg: Any,
        log_cfg: LogConfig,
        device: str,
        generate_target_image: bool = False,
    ) -> None:
        """Run pretrain phase for Neural painter.
        
        NeuralSVG pretrain flow:
        1. Generate target images if needed
        2. Load images and get attention maps for all phases
        3. Initialize painter with last phase's attention map (actual stroke positions)
        4. Compute target points from each phase's attention maps (pretrain targets)
        5. Save attention maps visualization
        6. Train MLP to predict offsets that move strokes toward target positions
        
        Args:
            phases: List of phase configurations
            data_cfg: Data configuration
            optim_cfg: Optimization configuration
            log_cfg: Logging configuration
            device: Device to use
            generate_target_image: Whether to generate target images
        """
        # Generate target images if needed
        if generate_target_image:
            print("\n------- Generating target images -------")
            self._generate_target_images(phases, data_cfg, device, log_cfg.setup_dir)
        
        # Load images and get attention maps for all phases
        phase_images = []
        phase_masks = []
        phase_image_tensors = []
        phase_attn_maps = []
        phase_input_images = []
        phase_attn_data = []  # Store full attention data for each phase
        
        print("\n------- Loading target images for all phases -------")
        for phase_idx, phase in enumerate(phases):
            # Use image_path from phase config if available, otherwise use generated image path
            if not phase.image_path or not Path(phase.image_path).exists():
                raise ValueError(f"Target image for {phase.name} not found: {phase.image_path}")
            
            print(f"  Loading {phase.name}: {phase.image_path}")
            image, mask, image_tensor = prepare_image(phase.image_path, data_cfg)
            input_image, attn_map = get_clip_attention_map(
                input_image=image,
                image_size=data_cfg.render_size,
                device=device
            )
            
            phase_images.append(image)
            phase_masks.append(mask)
            phase_image_tensors.append(image_tensor)
            phase_attn_maps.append(attn_map)
            phase_input_images.append(input_image)
        
        # Use last phase's image for painter initialization (needs all strokes)
        last_phase_idx = len(phases) - 1
        num_strokes_last = self.get_phase_stroke_count(last_phase_idx, phases)
        
        print(f"---- Initializing strokes with last phase ({phases[last_phase_idx].name}) attention positions ----")
        (
            attn_map_soft_list_last, bg_attn_map_soft_list_last,
            inds_last, inds_norm_last_full, bg_inds_last, bg_inds_norm_last
        ) = set_init_strokes_with_attention_map(
            attention_map=phase_attn_maps[last_phase_idx],
            input_image=phase_input_images[last_phase_idx],
            num_strokes=num_strokes_last,
            image_size=data_cfg.render_size,
            xdog_intersec=data_cfg.attn_init_xdog_intersec,
            mask=phase_masks[last_phase_idx],
            tau_max_min=tuple(data_cfg.attn_init_tau_max_min),
        )
        
        # Convert pixel indices to list of tuples [(h, w), ...] for color sampling
        pixel_inds_last = [(int(inds_last[i, 0]), int(inds_last[i, 1])) for i in range(len(inds_last))]
        
        # Initialize painter strokes with last phase's attention map
        self.init_strokes(
            phases=phases,
            attention_inds=inds_norm_last_full,
            pixel_inds=pixel_inds_last,
            target_image=phase_image_tensors[last_phase_idx],
        )
        
        # Save initial sketch before pretrain
        self.save_init_sketch("init_sketch_before_pretrain.jpg", log_cfg.pretrain_logs_dir)
        
        # Compute target points for each phase
        print("Computing pretrain target points for all phases...")
        points_init = []
        
        for phase_idx, phase in enumerate(phases):
            num_strokes = self.get_phase_stroke_count(phase_idx, phases)
            
            (
                attn_map_soft_list, bg_attn_map_soft_list,
                inds, inds_norm, bg_inds, bg_inds_norm
            ) = set_init_strokes_with_attention_map(
                attention_map=phase_attn_maps[phase_idx],
                input_image=phase_input_images[phase_idx],
                num_strokes=num_strokes,
                image_size=data_cfg.render_size,
                xdog_intersec=data_cfg.attn_init_xdog_intersec,
                mask=phase_masks[phase_idx],
                tau_max_min=tuple(data_cfg.attn_init_tau_max_min),
            )
            
            # Generate target points for this phase
            target_points = self._generate_target_points(inds_norm, num_strokes)
            points_init.append(target_points)
            
            phase_attn_data.append({
                'attn_map_soft_list': attn_map_soft_list,
                'bg_attn_map_soft_list': bg_attn_map_soft_list,
                'inds': inds,
                'bg_inds': bg_inds,
            })
        
        # Store points_init for pretrain loop
        self.points_init_pretrain = points_init
        
        # Prepare phase names
        phase_names = [phase.name for phase in phases]
        
        # Save all attention maps
        save_attention_maps(
            phase_attn_maps=phase_attn_maps,
            phase_images=phase_images,
            phase_attn_data=phase_attn_data,
            phase_names=phase_names,
            output_dir=log_cfg.setup_dir,
        )
        
        # Run pretrain loop
        self._pretrain_loop(phases, optim_cfg, log_cfg)
        
        # Save sketch after pretrain (before regular training)
        self.save_init_sketch("init_sketch_before_regular_training.jpg", log_cfg.pretrain_logs_dir)
    
    def _generate_target_points(self, attention_inds: List, num_strokes: int) -> torch.Tensor:
        """
        Generate target point positions from attention indices using NeuralSVG's method.
        
        This temporarily sets the painter's stroke_idxs to the target positions,
        then uses the painter's _get_shape/_get_path methods to generate points.
        This ensures the target points are generated exactly the same way as the
        painter initializes its strokes.
        """
        # Save original stroke_idxs
        original_stroke_idxs = self.stroke_idxs
        
        # Temporarily set stroke_idxs to target positions
        self.stroke_idxs = attention_inds
        
        target_points = []
        for idx in range(num_strokes):
            if self.is_closed:
                _, points = self._get_shape(idx)
            else:
                _, points = self._get_path(idx)
            target_points.append(points)
        
        # Restore original stroke_idxs
        self.stroke_idxs = original_stroke_idxs
        
        return torch.stack(target_points)
    
    def _generate_target_images(
        self,
        phases: List[PhaseConfig],
        data_cfg: Any,
        device: str,
        setup_dir: Path,
    ) -> None:
        """Generate target images for all phases using Stable Diffusion."""
        from diffusers import StableDiffusionPipeline, DDIMScheduler
        import torch
        # Check if all images already exist
        all_exist = True
        phase_paths = []
        for phase_idx, phase in enumerate(phases):
            # Use image_path from phase config if available, otherwise use generated image path
            if phase.image_path and Path(phase.image_path).exists():
                phase_path = Path(phase.image_path)
            else:
                phase_name = phase.name
                phase_path = setup_dir / f"generated_image_{phase_name}.png"
            phase_paths.append(phase_path)
            if not phase_path.exists():
                all_exist = False
        
        # Use existing images if all are available
        if all_exist:
            print(f"Using existing generated images:")
            for phase_idx, phase in enumerate(phases):
                print(f"  {phase.name}: {phase_paths[phase_idx]}")
                # Update phase config with the path
                phase.image_path = str(phase_paths[phase_idx])
            return
        
        # Load SD model
        print(f"Loading SD model for image generation: {self.sd_model}")
        
        pipe = StableDiffusionPipeline.from_pretrained(self.sd_model, torch_dtype=torch.float16)
        pipe.scheduler = DDIMScheduler.from_pretrained(self.sd_model, subfolder="scheduler")
        
        # Load LoRA if available
        if self.lora_weights:
            pipe.load_lora_weights(self.lora_weights)
        
        pipe = pipe.to(device)
        
        # Generate images for each phase
        for phase_idx, phase in enumerate(phases):
            # Use image_path from phase config if available, otherwise use generated image path
            if phase.image_path and Path(phase.image_path).exists():
                phase_path = Path(phase.image_path)
                print(f"Skipping {phase.name}: using existing image at {phase_path}")
                phase_paths[phase_idx] = phase_path
                continue
            
            # Generate new image path
            phase_name = phase.name
            phase_path = setup_dir / f"generated_image_{phase_name}.png"
            
            # Skip if already exists
            if phase_path.exists():
                print(f"Skipping {phase.name}: already exists at {phase_path}")
                phase.image_path = str(phase_path)
                phase_paths[phase_idx] = phase_path
                continue
            
            # Generate image for this phase
            print(f"Generating image for {phase.name}: '{phase.caption}'")
            generator = torch.Generator(device=device).manual_seed(getattr(data_cfg, 'seed', 42) + phase_idx)
            image = pipe(
                prompt=phase.caption,
                negative_prompt=getattr(data_cfg, 'negative_prompt', None),
                num_inference_steps=50,
                generator=generator,
            ).images[0]
            
            # Save directly to setup_dir
            image.save(phase_path)
            print(f"  Saved to: {phase_path}")
            # Update phase config with the path
            phase.image_path = str(phase_path)
            phase_paths[phase_idx] = phase_path
        
        # Clean up pipeline
        del pipe
        torch.cuda.empty_cache()
    
    def _pretrain_loop(
        self,
        phases: List[PhaseConfig],
        optim_cfg: OptimConfig,
        log_cfg: LogConfig,
    ) -> None:
        """
        Run pretrain loop with L2 loss for all phases.
        
        This follows NeuralSVG's pretrain logic:
        - For each toggle_color, compute L2 loss on point positions for all phases
        - Optional color pretrain loss
        - Accumulate gradients across all toggle_colors and phases
        """
        # Initialize optimizer for pretrain (use pretrain LR)
        # NeuralSVG uses AdamW with specific betas and eps
        pretrain_optimizer = torch.optim.AdamW(
            self.get_points_parameters(),
            lr=optim_cfg.learning_rate_pretrain,
            betas=(0.9, 0.9),  # NeuralSVG default
            eps=1e-6,
            weight_decay=getattr(optim_cfg, 'weight_decay', 0.0),
        )
        
        # Pretrain settings
        max_pretrain_steps = optim_cfg.max_steps_pretrain
        save_interval = log_cfg.save_interval
        
        # Get toggle colors (use all colors during pretrain)
        toggle_colors = [None]  # Default: no toggle color
        if self.toggle_color and self.toggle_color_bg_colors:
            toggle_colors = self.toggle_color_bg_colors
        
        print(f"Running pretrain for {max_pretrain_steps} steps with {len(toggle_colors)} toggle colors and {len(phases)} phases...")
        epoch_range = tqdm(range(max_pretrain_steps), desc="Pretrain")
        
        for step in epoch_range:
            pretrain_optimizer.zero_grad()
            
            # Accumulate loss across all toggle colors and phases
            total_l2_loss = 0.0
            total_color_loss = 0.0
            
            for toggle_color_value in toggle_colors:
                # Compute loss for each phase
                phase_l2_losses = []
                phase_color_losses = []
                phase_weights = []
                
                for phase_idx, phase in enumerate(phases):
                    num_strokes = self.get_phase_stroke_count(phase_idx, phases)
                    weight = phase.weight
                    
                    # Get predicted points for this phase
                    points, colors = self.get_points(
                        num_strokes=num_strokes,
                        toggle_color_value=toggle_color_value,
                    )
                    
                    # Remove batch dimension if present
                    if points.dim() == 4 and points.shape[0] == 1:
                        points = points.squeeze(0)  # [N, num_points, 2]
                    
                    # L2 loss directly on canvas coordinates
                    l2_loss = torch.nn.functional.mse_loss(points, self.points_init_pretrain[phase_idx])
                    phase_l2_losses.append(weight * l2_loss)
                    phase_weights.append(weight)
                    
                    # Optional color pretrain loss
                    color_loss = torch.tensor(0.0, device=self.device)
                    if self.use_color and colors is not None:
                        # Get ground truth colors from initial colors
                        gt_colors = None
                        if self.toggle_color and hasattr(self, 'shapes_init_colors'):
                            if isinstance(self.shapes_init_colors, dict):
                                gt_colors = self.shapes_init_colors.get(
                                    toggle_color_value,
                                    self.shapes_init_colors.get(toggle_colors[0])
                                )
                            else:
                                gt_colors = self.shapes_init_colors
                        elif hasattr(self, 'shapes_init_colors'):
                            gt_colors = self.shapes_init_colors
                        
                        if gt_colors is not None:
                            gt_colors_phase = gt_colors[:num_strokes]  # (num_strokes, 4)
                            color_loss = torch.nn.functional.mse_loss(colors, gt_colors_phase)
                            phase_color_losses.append(weight * color_loss)
                        else:
                            phase_color_losses.append(torch.tensor(0.0, device=self.device))
                    else:
                        phase_color_losses.append(torch.tensor(0.0, device=self.device))
                
                # Combined loss across all phases
                total_weight = sum(phase_weights)
                if total_weight > 0:
                    l2_loss = sum(phase_l2_losses) / total_weight
                    color_loss = sum(phase_color_losses) / total_weight
                else:
                    l2_loss = sum(phase_l2_losses)
                    color_loss = sum(phase_color_losses)
                
                # Backward for this toggle color
                loss = l2_loss + color_loss
                loss.backward()
                
                total_l2_loss += l2_loss.item()
                total_color_loss += color_loss.item()
            
            # Average loss for logging
            avg_l2_loss = total_l2_loss / len(toggle_colors)
            avg_color_loss = total_color_loss / len(toggle_colors)
            
            # Clip gradients
            if getattr(optim_cfg, 'use_clip_grad', False):
                torch.nn.utils.clip_grad_norm_(
                    self.get_points_parameters(),
                    max_norm=getattr(optim_cfg, 'clip_grad_max_norm_points', 1.0),
                )
            
            pretrain_optimizer.step()
            
            # Update progress
            epoch_range.set_postfix(
                l2=f"{avg_l2_loss:.6f}",
                color=f"{avg_color_loss:.6f}"
            )
            
            # Early termination (like NeuralSVG: l2_loss < 3.0, color_loss < 0.02)
            l2_threshold = 3.0
            if avg_l2_loss < l2_threshold:
                color_ok = (not self.use_color) or (avg_color_loss < 0.02)
                if color_ok:
                    print(f"Pretrain converged at step {step} (l2={avg_l2_loss:.6f}, color={avg_color_loss:.6f})")
                    break
            
            # Save intermediate results
            if step % save_interval == 0:
                self._save_pretrain_checkpoint(step, phases, log_cfg)
        
        # Save final pretrain checkpoint
        self._save_pretrain_checkpoint(max_pretrain_steps, phases, log_cfg)
        print("Pretrain completed")
    
    def _save_pretrain_checkpoint(
        self,
        step: int,
        phases: List[PhaseConfig],
        log_cfg: LogConfig,
    ) -> None:
        """Save pretrain checkpoint with toggle colors for all phases."""
        pretrain_logs_dir = log_cfg.pretrain_logs_dir
        
        # Get toggle colors for Neural painter
        toggle_colors = ["white"]  # Default
        if self.toggle_color and self.toggle_color_bg_colors:
            toggle_colors = self.toggle_color_bg_colors
        
        num_phases = len(phases)
        num_colors = len(toggle_colors)
        
        try:
            # Store images: all_images[color_idx][phase_idx] = image_pil
            all_images = [[None for _ in range(num_phases)] for _ in range(num_colors)]
            
            # Set model to eval mode for rendering
            if self.mlp is not None:
                self.mlp.eval()
            
            with torch.no_grad():
                for color_idx, toggle_color in enumerate(toggle_colors):
                    for phase_idx, phase in enumerate(phases):
                        num_strokes = self.get_phase_stroke_count(phase_idx, phases)
                        
                        # Render with this toggle color
                        img = self.get_image(
                            truncation_indices=[num_strokes],
                            toggle_color_value=toggle_color,
                        )

                        all_images[color_idx][phase_idx] = tensor_to_pil(img)
            
            # Set model back to train mode
            if self.mlp is not None:
                self.mlp.train()
            
            # Create combined jpg_logs image (grid of all phases and toggle colors)
            if num_colors > 1 or num_phases > 1:
                self._save_pretrain_grid_image(
                    all_images, step, pretrain_logs_dir, toggle_colors, phases
                )
            
        except Exception as e:
            print(f"Warning: Could not save pretrain checkpoint: {e}")
    
    def _save_pretrain_grid_image(
        self,
        all_images: List[List[Image.Image]],  # [color_idx][phase_idx]
        step: int,
        output_dir: Path,
        toggle_colors: List[str],
        phases: List[PhaseConfig],
    ) -> None:
        """Save a grid image showing all phases and toggle colors for pretrain visualization."""
        num_colors = len(all_images)
        num_phases = len(all_images[0]) if all_images else 0
        
        # Create grid: rows = num_phases, cols = num_colors
        fig, axes = plt.subplots(num_phases, num_colors, figsize=(3 * num_colors, 3 * num_phases))
        
        # Handle single phase or single color case
        if num_phases == 1:
            axes = axes.reshape(1, -1)
        if num_colors == 1:
            axes = axes.reshape(-1, 1)
        
        # Plot images in grid: outer loop for phases, inner loop for colors
        for phase_idx, phase in enumerate(phases):
            n_strokes = self.get_phase_stroke_count(phase_idx, phases)
            
            for color_idx, toggle_color in enumerate(toggle_colors):
                ax = axes[phase_idx, color_idx]
                img_pil = all_images[color_idx][phase_idx]
                
                ax.imshow(img_pil)
                ax.axis('off')
                
                # Set title: color name on top row
                if phase_idx == 0:
                    ax.set_title(f"{toggle_color}", fontsize=10)
                
                # Set ylabel: phase info on first column
                if color_idx == 0:
                    title = f"{phase.name}\n({n_strokes} strokes)"
                    ax.set_ylabel(title, fontsize=10, rotation=0, ha='right', va='center')
        
        fig.suptitle(f"Pretrain Step {step}", fontsize=14)
        plt.tight_layout()
        
        plt.savefig(output_dir / f"iter_{step}.jpg", dpi=100, bbox_inches='tight')
        plt.close(fig)
    
    def save_init_sketch(self, filename: str, output_dir: str) -> None:
        """Save initial sketch image with toggle colors.
        
        Args:
            filename: Filename to save the sketch
            output_dir: Output directory where the sketch will be saved
        """
        pretrain_logs_dir = Path(output_dir)
        pretrain_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Get toggle colors
        toggle_colors = ["white"]
        if self.toggle_color and self.toggle_color_bg_colors:
            toggle_colors = self.toggle_color_bg_colors
        
        try:
            # Use first toggle color for the main image
            toggle_color = toggle_colors[0] if toggle_colors else None
            
            # Render with all strokes
            if self.mlp is not None:
                self.mlp.eval()
            with torch.no_grad():
                img = self.get_image(
                    num_strokes=self.num_strokes,
                    toggle_color_value="white",
                    render_without_background=True,
                )
            if self.mlp is not None:
                self.mlp.train()
            
            # Convert to PIL and save
            tensor_to_pil(img, save_path=pretrain_logs_dir / filename)
        except Exception as e:
            print(f"Warning: Could not save init sketch: {e}")


# =============================================================================
# NeuralPainter Optimizer
# =============================================================================

class NeuralPainterOptimizer:
    """
    Optimizer wrapper for NeuralPainter using AdamW with NeuralSVG settings.
    
    Uses the same optimizer configuration as NeuralSVG's coach.py:
    - AdamW with betas=(0.9, 0.9), eps=1e-6
    - Supports separate pretrain and regular training learning rates
    - Custom learning rate schedulers (sinusoidal ramp, cosine decay, etc.)
    """

    def __init__(
        self,
        painter: "NeuralPainter",
        lr: float = 0.018,
        lr_pretrain: float = 0.01,
        weight_decay: float = 0.0,
        pretrain: bool = False,
    ):
        """
        Initialize optimizer for NeuralPainter.
        
        Args:
            painter: The NeuralPainter instance to optimize.
            lr: Learning rate for regular training.
            lr_pretrain: Learning rate for pretraining.
            weight_decay: Weight decay for AdamW.
            pretrain: Whether in pretrain mode.
        """
        self.painter = painter
        self.lr = lr
        self.lr_pretrain = lr_pretrain
        self.pretrain = pretrain
        
        # Get parameters from painter
        params = list(painter.get_parameters())
        
        # Use pretrain or regular learning rate
        current_lr = lr_pretrain if pretrain else lr
        
        # AdamW with NeuralSVG settings
        self.optimizer = torch.optim.AdamW(
            params,
            lr=current_lr,
            betas=(0.9, 0.9),  # NeuralSVG default
            eps=1e-6,          # NeuralSVG default
            weight_decay=weight_decay,
        )
        self.scheduler = None
        self._scheduler_type = None
        self._max_steps = None

    def zero_grad(self) -> None:
        """Zero gradients."""
        self.optimizer.zero_grad()

    def step(self) -> None:
        """Perform optimization step."""
        self.optimizer.step()
        
        if self.scheduler is not None:
            self.scheduler.step()

    def set_scheduler(
        self,
        scheduler_type: str,
        max_steps: int,
        lr_init: Optional[float] = None,
        lr_final: float = 0.0,
        warmup_steps: int = 100,
        lr_delay_mult: float = 0.1,
    ) -> None:
        """
        Set learning rate scheduler with NeuralSVG-style options.
        
        Args:
            scheduler_type: Type of scheduler:
                - "sinusoidal_ramp_exponential_decay"
                - "linear_ramp_cosine_decay" 
                - "exp_ramp_cosine_decay"
                - "cosine", "linear", "constant"
            max_steps: Total number of optimization steps.
            lr_init: Initial learning rate (default: use optimizer's lr).
            lr_final: Final learning rate for decay schedulers.
            warmup_steps: Number of warmup steps.
            lr_delay_mult: Multiplier during delay/warmup phase.
        """
        self._scheduler_type = scheduler_type
        self._max_steps = max_steps

        if lr_init is None:
            lr_init = self.lr_pretrain if self.pretrain else self.lr
        
        if scheduler_type in ["sinusoidal_ramp_exponential_decay", "custom"]:
            lr_lambda = _NeuralSVGLRLambda(
                func_type="sinusoidal_ramp_exponential_decay",
                max_steps=max_steps,
                lr_init=lr_init,
                lr_final=lr_final,
                lr_delay_steps=warmup_steps,
                lr_delay_mult=lr_delay_mult,
            )
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lr_lambda
            )
        
        elif scheduler_type in ["linear_ramp_cosine_decay", "exp_ramp_cosine_decay"]:
            lr_lambda = _NeuralSVGLRLambda(
                func_type=scheduler_type,
                max_steps=max_steps,
                lr_init=lr_init,
                lr_final=lr_final,
                lr_delay_steps=warmup_steps,
                lr_delay_mult=lr_delay_mult,
            )
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lr_lambda
            )
        
        elif scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max_steps
            )
        
        elif scheduler_type == "linear":
            def linear_lambda(step):
                return 1.0 - step / max_steps
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=linear_lambda
            )
        
        else:
            # constant or unknown - no scheduler
            # self.scheduler = None
            self.scheduler = get_scheduler(
                scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=max_steps,
            )

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state dictionary."""
        state = {
            'optimizer': self.optimizer.state_dict(),
            'pretrain': self.pretrain,
        }
        if self.scheduler is not None:
            state['scheduler'] = self.scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load optimizer state from dictionary."""
        self.optimizer.load_state_dict(state_dict['optimizer'])
        if 'pretrain' in state_dict:
            self.pretrain = state_dict['pretrain']
        if self.scheduler is not None and 'scheduler' in state_dict:
            self.scheduler.load_state_dict(state_dict['scheduler'])


class _NeuralSVGLRLambda:
    """
    Learning rate scheduler lambda for NeuralSVG-style scheduling.
    
    Provides various learning rate scheduling functions:
    - Sinusoidal ramp with exponential decay
    - Linear ramp with cosine decay
    - Exponential ramp with cosine decay
    """

    def __init__(
        self,
        *,
        func_type: str,
        max_steps: int,
        lr_init: float,
        lr_final: float,
        lr_delay_steps: int = 100,
        lr_delay_mult: float = 0.1,
    ):
        """
        Initialize the learning rate scheduler.

        Args:
            func_type: Type of scheduling function.
            max_steps: Maximum number of steps.
            lr_init: Initial learning rate.
            lr_final: Final learning rate.
            lr_delay_steps: Steps before starting decay.
            lr_delay_mult: Multiplier during delay.
        """
        self.func_type = func_type
        self.max_steps = max_steps
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.lr_delay_steps = lr_delay_steps
        self.lr_delay_mult = lr_delay_mult

    def _sinusoidal_ramp_exponential_decay(self, step: int) -> float:
        """Sinusoidal ramp with exponential decay."""
        if step < self.lr_delay_steps:
            mult = self.lr_delay_mult + (1 - self.lr_delay_mult) * (
                np.sin(0.5 * np.pi * step / self.lr_delay_steps)
            )
        else:
            mult = np.exp(
                (step - self.lr_delay_steps)
                * np.log(self.lr_final)
                / (self.max_steps - self.lr_delay_steps)
            )
        return self.lr_init * mult

    def _linear_ramp_cosine_decay(self, step: int) -> float:
        """Linear ramp with cosine decay."""
        if step < self.lr_delay_steps:
            mult = (
                self.lr_delay_mult
                + (1 - self.lr_delay_mult) * step / self.lr_delay_steps
            )
        else:
            t = (step - self.lr_delay_steps) / (self.max_steps - self.lr_delay_steps)
            mult = np.cos(t * np.pi / 2) ** 2
        return self.lr_init * mult

    def _exp_ramp_cosine_decay(self, step: int) -> float:
        """Exponential ramp with cosine decay."""
        if step < self.lr_delay_steps:
            mult = self.lr_delay_mult + (1 - self.lr_delay_mult) * (
                np.sin(0.5 * np.pi * step / self.lr_delay_steps)
            )
        else:
            t = (step - self.lr_delay_steps) / (self.max_steps - self.lr_delay_steps)
            mult = np.cos(t * np.pi / 2) ** 2
        return self.lr_init * mult

    def __call__(self, step: int) -> float:
        """Calculate learning rate multiplier for current step."""
        if self.func_type == "sinusoidal_ramp_exponential_decay":
            return self._sinusoidal_ramp_exponential_decay(step)
        elif self.func_type == "linear_ramp_cosine_decay":
            return self._linear_ramp_cosine_decay(step)
        elif self.func_type == "exp_ramp_cosine_decay":
            return self._exp_ramp_cosine_decay(step)
        else:
            raise ValueError(f"Invalid function type: {self.func_type}")
