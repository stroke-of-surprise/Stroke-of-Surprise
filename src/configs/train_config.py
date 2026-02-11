"""Main training configuration combining all sub-configs."""

from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import os
import termcolor

from .data_config import DataConfig, PhaseConfig
from .optim_config import OptimConfig
from .log_config import LogConfig
from .painter_config import BezierConfig, BsplineConfig, NeuralConfig



class PainterType(Enum):
    """Supported painter types."""
    BEZIER = "bezier"
    BSPLINE = "bspline"
    NEURAL = "neural"


@dataclass
class TrainConfig:
    """Main training configuration for illusion sketch generation."""

    # Painter type selection
    painter_type: str = field(default="bezier")
    
    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    log: LogConfig = field(default_factory=LogConfig)
    
    # Painter-specific configurations
    bezier: BezierConfig = field(default_factory=BezierConfig)
    bspline: BsplineConfig = field(default_factory=BsplineConfig)
    neural: NeuralConfig = field(default_factory=NeuralConfig)
    
    # Training steps
    num_iter: int = field(default=2000)
    seed: int = field(default=0)
    num_of_samples: int = field(default=1)
    
    # Device
    use_cpu: bool = field(default=False)
    
    # Diffusion model
    sd_version: str = field(default="1.5")  # "1.5", "2.0", "2.1"
    
    # Phases (populated during post_init)
    phases: Optional[List[PhaseConfig]] = field(default=None)
    num_phases: int = field(default=0)


    def __post_init__(self) -> None:
        """Validate and initialize configuration."""
        # Validate painter type
        try:
            self.painter_type_enum = PainterType(self.painter_type)
        except ValueError:
            valid_types = [t.value for t in PainterType]
            raise ValueError(
                f"Invalid painter_type: {self.painter_type}. "
                f"Must be one of {valid_types}"
            )
        
        # Validate seed
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")
        
        # Validate num_of_samples
        if self.num_of_samples < 1:
            raise ValueError(f"num_of_samples must be at least 1, got {self.num_of_samples}")
        
        # Validate num_iter
        if self.num_iter <= 0:
            raise ValueError(f"num_iter must be positive, got {self.num_iter}")
        
        # Initialize phases configuration
        self._init_phases()
        
        # Set run name if not provided
        if not self.log.run_name:
            self._generate_run_name()
        
        # Update visualization indices based on phases
        if self.phases and len(self.phases) >= 2:
            self.log.visualization_truncation_idxs = [
                ph.num_strokes for ph in self.phases
            ]

    def _init_phases(self) -> None:
        """Initialize multi-phase configuration from various sources."""
        if self.data.phases_config:
            # Load from JSON file
            assert os.path.isfile(self.data.phases_config), \
                f"{self.data.phases_config} does not exist!"
            with open(self.data.phases_config, "r") as f:
                phases = json.load(f)
            assert isinstance(phases, list) and len(phases) >= 2, \
                "phases_config must be a list with at least 2 phases"
            
            # Convert dictionaries to PhaseConfig objects
            normalized_phases = []
            for idx, ph in enumerate(phases):
                assert isinstance(ph, dict), f"Phase {idx} must be a dict"
                phase_config = PhaseConfig(
                    name=ph.get("name", f"phase_{idx + 1}"),
                    caption=ph.get("caption", ""),
                    num_strokes=ph.get("num_strokes"),
                    weight=ph.get("weight", 1.0),
                    image_path=ph.get("image_path", None),
                )
                normalized_phases.append(phase_config)
            
            self.phases = normalized_phases
            
        elif self.data.caption_A and self.data.caption_B:
            # Legacy A/B mode
            phases = []
            
            # Phase A
            phase_A = PhaseConfig(
                name="A",
                weight=self.data.A_weight,
                caption=self.data.caption_A,
                num_strokes=self.data.split_stroke_num or self.data.num_strokes // 2,
                image_path=self.data.image_path_A,
            )
            phases.append(phase_A)
            
            # Phase B
            phase_B = PhaseConfig(
                name="B",
                weight=self.data.B_weight,
                caption=self.data.caption_B,
                num_strokes=self.data.num_strokes,
                image_path=self.data.image_path_B,
            )
            phases.append(phase_B)
            
            self.phases = phases
        else:
            # Single phase mode (no illusion)
            self.phases = [PhaseConfig(
                name="main",
                weight=1.0,
                caption=self.data.text_prompt,
                num_strokes=self.data.num_strokes,
                image_path=self.data.image_path,
            )]
        
        # Validate and normalize phases
        self.validate_phases()
        self.num_phases = len(self.phases)

    def validate_phases(self) -> None:
        """Validate and normalize phase configuration."""
        if not self.phases:
            return
        
        prev_num_strokes = 0
        for idx, ph in enumerate(self.phases):
            phase_id = idx + 1
            phase_name = ph.name
            
            # Validate caption
            if not ph.caption:
                raise ValueError(
                    f"Phase {phase_id} ({phase_name}) must have 'caption'"
                )
            
            # Handle num_strokes
            if ph.num_strokes is None:
                if phase_id == len(self.phases):
                    ph.num_strokes = self.data.num_strokes
                else:
                    raise ValueError(
                        f"Phase {phase_id} ({phase_name}) must have 'num_strokes'"
                    )
            
            # Validate ordering
            num_strokes = int(ph.num_strokes)
            if num_strokes < prev_num_strokes:
                raise ValueError(
                    f"Phase {phase_id} ({phase_name}) num_strokes ({num_strokes}) "
                    f"must be >= previous phase ({prev_num_strokes})"
                )
            prev_num_strokes = num_strokes
        
        # Update total num_strokes from last phase
        if self.phases:
            self.data.num_strokes = self.phases[-1].num_strokes

    def _generate_run_name(self) -> None:
        """Generate run name from phases."""
        if self.data.phases_config:
            self.log.run_name = Path(self.data.phases_config).stem
        elif len(self.phases) >= 2:
            names = [ph.caption.replace(" ", "_")[:20] for ph in self.phases[:2]]
            self.log.run_name = "+".join(names)
        else:
            self.log.run_name = self.phases[0].caption.replace(" ", "_")[:30]

    def get_painter_config(self):
        """Get the appropriate painter config based on painter_type."""
        if self.painter_type == "bezier":
            return self.bezier
        elif self.painter_type == "bspline":
            return self.bspline
        elif self.painter_type == "neural":
            return self.neural
        else:
            raise ValueError(f"Unknown painter type: {self.painter_type}")

    def print_config(self) -> None:
        """Print configuration summary."""
        print(f"\n{'='*60}")
        print(f"Training Configuration")
        print(f"{'='*60}")
        print(termcolor.colored(f"Painter Type: {self.painter_type}", "blue"))
        print(f"Iterations: {self.num_iter}")
        print(f"Total Strokes: {self.data.num_strokes}")
        
        print(f"\nPhases ({self.num_phases}):")
        for ph in self.phases:
            print(termcolor.colored(f"  {ph.name}: {ph.num_strokes} strokes, "
                  f"weight={ph.weight}, caption='{ph.caption[:50]}...'", "yellow"))
        
        print(termcolor.colored(f"\nOutput: {self.log.exp_dir}", "blue"))
        print(f"{'='*60}\n")

