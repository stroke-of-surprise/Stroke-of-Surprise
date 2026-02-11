"""Optimizer and learning rate scheduler configuration."""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Union


class SchedulerType(Enum):
    """Learning rate scheduler types."""
    COSINE = "cosine"
    STEP = "step"
    CONSTANT = "constant"
    LINEAR = "linear"
    DREAMTIME = "dreamtime"
    LIN_RAMP_COS_DECAY = "linear_ramp_cosine_decay"
    EXP_RAMP_COS_DECAY = "exp_ramp_cosine_decay"
    SIN_RAMP_EXP_DECAY = "sinusoidal_ramp_exponential_decay"


@dataclass
class OptimConfig:
    """Optimization configuration for training."""

    # Learning rates
    learning_rate: float = field(default=0.8)  # BezierSketch default
    learning_rate_color: float = field(default=0.01)
    learning_rate_pretrain: float = field(default=1e-2)  # NeuralSVG pretrain LR
    target_lr: float = field(default=0.012)  # Target LR for scheduler decay
    
    # Scheduler
    scheduler_type: Union[SchedulerType, str] = field(default=SchedulerType.CONSTANT)
    scheduler_type_pretrain: Union[SchedulerType, str] = field(default=SchedulerType.CONSTANT)
    
    warmup_steps: int = field(default=50)
    
    # Gradient clipping
    use_clip_grad: bool = field(default=True)
    clip_grad_max_norm: float = field(default=0.1)
    clip_grad_max_norm_colors: float = field(default=0.1)  # NeuralSVG: separate for colors
    clip_grad_max_norm_points: float = field(default=0.1)  # NeuralSVG: separate for points
    
    # Weight decay
    weight_decay: float = field(default=0.0)
    
    # SDS configuration (BezierSketch defaults)
    sds_guidance_scale: float = field(default=100.0)  # BezierSketch: 100, BsplineSketch: 7.5
    sds_grad_scale: float = field(default=10.0)  # BezierSketch default
    sds_t_range_min: float = field(default=0.05)  # BezierSketch: 0.05, BsplineSketch: 0.5
    sds_t_range_max: float = field(default=0.95)  # BezierSketch: 0.95, BsplineSketch: 0.98
    sds_time_schedule: str = field(default="random")  # BezierSketch: "random", BsplineSketch: "ism"
    sds_grad_method: str = field(default="sds")  # BezierSketch: "sds", BsplineSketch: "ism"
    sds_weight: str = field(default="default")  # NeuralSVG: "default", "constant", etc.
    sds_loss_style: str = field(default="mse")  # NeuralSVG: "mse", "simple"

    # Overlay loss
    overlay_loss_type: str = field(default="dot")
    overlay_loss_weight: float = field(default=5.0)
    overlay_blur_sigma: float = field(default=2.0)
    overlay_blur_kernel_size: int = field(default=15)
    overlay_hinge_threshold: float = field(default=0.5)
    
    # Pretrain (for Neural painter)
    pretrain: bool = field(default=False)
    lambda_l2_pretraining: float = field(default=1.0)
    max_steps_pretrain: int = field(default=400)
    timestep_scheduling: Optional[str] = field(default=None)
    sds_sample_timestep_sd: float = field(default=100)
    sd_num_inference_steps: int = field(default=50)
    sds_use_bg_color_suffix: bool = field(default=True)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )

        if self.learning_rate_pretrain <= 0:
            raise ValueError(
                f"learning_rate_pretrain must be positive, got {self.learning_rate_pretrain}"
            )

        if self.warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be non-negative, got {self.warmup_steps}"
            )

        if self.weight_decay < 0:
            raise ValueError(
                f"weight_decay must be non-negative, got {self.weight_decay}"
            )
        
        if self.overlay_loss_type not in ["None", "dot", "iou", "dice", "hinge"]:
            raise ValueError(
                f"overlay_loss_type must be one of 'None', 'dot', 'iou', 'dice', 'hinge', got {self.overlay_loss_type}"
            )
            
        if self.overlay_loss_weight < 0:
            raise ValueError(
                f"overlay_loss_weight must be non-negative, got {self.overlay_loss_weight}"
            )

