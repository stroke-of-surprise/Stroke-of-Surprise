"""Data configuration for stroke generation."""

from typing import Optional, List
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PhaseConfig:
    """Configuration for a single phase in multi-phase training."""
    
    name: str = field(default="phase_1")
    caption: str = field(default="")
    num_strokes: Optional[int] = field(default=None)
    weight: float = field(default=1.0)
    image_path: Optional[str] = field(default=None)  # Path to target image for this phase (for pretrain)


@dataclass
class DataConfig:
    """Configuration for training data and stroke parameters."""

    # Prompt configuration
    text_prompt: str = field(default="A sketch")
    text_prompt_suffix: str = field(
        default=""
        # default="minimal 2d line drawing. on a white background."
    )
    negative_prompt: str = field(
        default=""
        # default="unrealistic, blurry, low quality, out of focus, ugly, low contrast, dull, dark, low-resolution, gloomy"
    )
    
    # Stroke configuration
    num_strokes: int = field(default=32)
    num_segments: int = field(default=1)
    control_points_per_seg: int = field(default=4)
    width: float = field(default=1.5)
    
    # Rendering
    render_size: int = field(default=512)
    output_svg_size: int = field(default=512)
    
    # Optional inputs
    init_strokes_svg: Optional[str] = field(default=None)
    background_image: Optional[str] = field(default=None)
    target_image: Optional[str] = field(default=None)
    
    # Color optimization
    optimize_color: bool = field(default=False)
    background_color: str = field(default="white")
    stroke_color: str = field(default="black")
    
    # For Neural painter
    is_closed: bool = field(default=True)
    radius: float = field(default=0.05)
    regular_polygon_closed_shape_init: bool = field(default=True)
    use_background: bool = field(default=True)  # Enable background rendering for toggle colors
    
    # Multi-phase / Illusion sketch configuration
    use_illusion_sketch: bool = field(default=True)
    phases_config: Optional[str] = field(default=None)
    
    # Legacy A/B mode (converted to phases internally)
    caption_A: Optional[str] = field(default=None)
    caption_B: Optional[str] = field(default=None)
    split_stroke_num: Optional[int] = field(default=None)
    A_weight: float = field(default=1.0)
    B_weight: float = field(default=1.0)
    
    # Target image generation (for Neural painter pretrain)
    generate_target_image: bool = field(default=True)
    image_path: Optional[str] = field(default=None)
    image_path_A: Optional[str] = field(default=None)  # Target image for phase A (pretrain)
    image_path_B: Optional[str] = field(default=None)  # Target image for phase B (pretrain)
    text_prompt_A: Optional[str] = field(default=None)  # Text prompt for generating image A
    text_prompt_B: Optional[str] = field(default=None)  # Text prompt for generating image B
    
    # Attention map initialization parameters
    attn_init_tau_max_min: List[float] = field(default_factory=lambda: [0.3, 0.3])
    attn_init_xdog_intersec: bool = field(default=True)
    segment_object: bool = field(default=False)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.num_strokes <= 0:
            raise ValueError(f"num_strokes must be positive, got {self.num_strokes}")

        if self.num_segments <= 0:
            raise ValueError(f"num_segments must be positive, got {self.num_segments}")

        if self.render_size <= 0:
            raise ValueError(f"render_size must be positive, got {self.render_size}")

        if self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}")

        if self.control_points_per_seg <= 0:
            raise ValueError(
                f"control_points_per_seg must be positive, got {self.control_points_per_seg}"
            )
