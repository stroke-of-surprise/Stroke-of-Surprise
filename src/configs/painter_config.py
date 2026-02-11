"""Painter configuration for different painter types."""

from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field


@dataclass
class BezierConfig:
    """Configuration specific to Bezier painter."""
    
    # Control points
    control_points_per_seg: int = field(default=4)
    num_segments: int = field(default=1)
    
    # Stroke width
    width: float = field(default=2.5)
    width_optim: bool = field(default=False)

    def __post_init__(self) -> None:
        if self.control_points_per_seg < 2:
            raise ValueError(
                f"control_points_per_seg must be at least 2, got {self.control_points_per_seg}"
            )


@dataclass
class BsplineConfig:
    """Configuration specific to B-spline painter."""
    
    # B-spline parameters
    spline_degree: int = field(default=5)
    multiplicity: int = field(default=3)  # BsplineSketch default: 3
    num_control_points: int = field(default=5)  # BsplineSketch default: 5

    # Initialization mode
    init_mode: str = field(default="circular")  # "circular", "grid", "random"
    
    # Stroke width
    width: float = field(default=2.5)
    min_width: float = field(default=1.0)
    max_width: float = field(default=8.0)
    width_optim: bool = field(default=True)  # BsplineSketch: vary_stroke_width=1
    
    # Learning rates
    lr_pos: float = field(default=3.0)  # Learning rate for control points
    lr_width: float = field(default=0.5)  # Learning rate for stroke widths
    
    # Smoothing loss
    smoothing_weight: float = field(default=200.0)  # BsplineSketch default: 200.0
    smoothing_deriv: int = field(default=3)  # 3=jerk, 4=snap
    use_pspline: bool = field(default=False)

    def __post_init__(self) -> None:
        if self.spline_degree < 1:
            raise ValueError(
                f"spline_degree must be at least 1, got {self.spline_degree}"
            )
        if self.multiplicity < 1:
            raise ValueError(
                f"multiplicity must be at least 1, got {self.multiplicity}"
            )


@dataclass
class NeuralConfig:
    """Configuration specific to Neural (MLP) painter."""
    
    # MLP architecture
    mlp_dim: int = field(default=128)
    mlp_num_layers: int = field(default=2)
    input_dim: int = field(default=128)
    
    # Color prediction
    use_color: bool = field(default=True)
    
    # Nested dropout
    use_nested_dropout: bool = field(default=True)
    truncation_start_idx: int = field(default=4)
    nested_dropout_sampling_method: str = field(default="uniform")
    start_nested_dropout_from_step: int = field(default=200)
    
    # Dropout value prediction
    use_dropout_value: bool = field(default=False)
    dropout_emb_dim: int = field(default=16)
    dropout_last_item_prob: float = field(default=0.5)  # NeuralSVG default: 0.5
    dropout_temperature: float = field(default=1.5)  # NeuralSVG default: 1.5
    
    # Point prediction scale
    points_prediction_scale: float = field(default=0.1)
    
    # Color toggle
    toggle_color: bool = field(default=True)
    toggle_color_method: str = field(default="rgb")
    toggle_color_input_dim: int = field(default=8)
    toggle_color_bg_colors: List[str] = field(default_factory=lambda: ["blue", "red"])
    toggle_color_init_eps: float = field(default=0.1)
    toggle_sample_random_color_prob: float = field(default=0.5)
    
    # Aspect ratio toggle
    toggle_aspect_ratio: bool = field(default=False)
    toggle_aspect_ratio_values: List[str] = field(default_factory=lambda: ["1:1"])
    aspect_ratio_emb_dim: int = field(default=16)
    
    # LoRA weights for SD
    lora_weights: Optional[str] = field(default=None)
    
    # Checkpoint path
    mode: str = field(default="train")  # "train", "eval", "inference"
    checkpoint_path: Optional[Path] = field(default=None)

    # Target image generation
    generate_target_image: bool = field(default=True)
    attn_init_tau_max_min: List[float] = field(default_factory=lambda: [0.3, 0.3])
    attn_init_xdog_intersec: bool = field(default=True)
    
    # Text-to-image model for target image generation
    text2img_model: str = field(default="stabilityai/stable-diffusion-2-1")

    def __post_init__(self) -> None:
        if not self.toggle_color:
            self.toggle_color_bg_colors = [None]
        else:
            # Clean up color background names
            self.toggle_color_bg_colors = list(
                map(
                    lambda x: x.replace("-", " ") if x is not None else x,
                    self.toggle_color_bg_colors,
                )
            )

        if self.nested_dropout_sampling_method not in ["uniform", "exp_decay"]:
            raise ValueError(
                f"unsupported value {self.nested_dropout_sampling_method=}"
            )
            
        if self.mlp_dim <= 0:
            raise ValueError(f"mlp_dim must be positive, got {self.mlp_dim}")
            
        if self.mlp_num_layers <= 0:
            raise ValueError(f"mlp_num_layers must be positive, got {self.mlp_num_layers}")

