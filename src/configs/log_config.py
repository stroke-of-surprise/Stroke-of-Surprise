"""Logging configuration for training."""

from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class LogConfig:
    """Configuration for logging and output."""

    # Output directories
    output_dir: Path = field(default=Path("./output"))
    run_name: str = field(default="")

    # Logging intervals
    save_interval: int = field(default=100)
    
    # Visualization
    visualization_truncation_idxs: List[int] = field(
        default_factory=lambda: [64, 32, 16, 8, 4, 2]
    )
    
    # Allow overwrite
    allow_overwrite: bool = field(default=False)

    def __post_init__(self) -> None:
        """Validate and create directories."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
            
        if self.save_interval <= 0:
            raise ValueError(
                f"save_interval must be positive, got {self.save_interval}"
            )

    @property
    def exp_dir(self) -> Path:
        """Get experiment directory."""
        return self.output_dir / self.run_name
    
    @property
    def svg_logs_dir(self) -> Path:
        """Get SVG logs directory."""
        return self.exp_dir / "svg_logs"
    
    @property
    def png_logs_dir(self) -> Path:
        """Get PNG logs directory (for Bezier/Bspline)."""
        return self.exp_dir / "png_logs"
    
    @property
    def pretrain_logs_dir(self) -> Path:
        """Get pretrain logs directory (for Neural)."""
        return self.exp_dir / "pretrain_logs"
    
    @property
    def setup_dir(self) -> Path:
        """Get setup directory for initial configurations."""
        return self.exp_dir / "setup"

