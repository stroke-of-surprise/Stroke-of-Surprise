"""
Abstract base class for all painter implementations.

This module defines the common interface that all painters (Bezier, B-spline, Neural)
must implement for the IllusionTrainer to work uniformly with different stroke
representations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from ..utils.sketch_utils import tensor_to_pil
from ..configs.data_config import PhaseConfig


class BasePainter(ABC):
    """
    Abstract base class for stroke-based painters.
    
    All painter implementations (Bezier, B-spline, Neural) must inherit from this
    class and implement the required abstract methods.
    
    The painter is responsible for:
    - Managing stroke parameters (control points, widths, colors)
    - Rendering strokes to images
    - Providing parameters for optimization
    - Saving/loading stroke representations (SVG, etc.)
    """

    def __init__(
        self,
        num_strokes: int,
        canvas_size: int = 512,
        device: torch.device = None,
    ):
        """
        Initialize the base painter.
        
        Args:
            num_strokes: Total number of strokes to generate.
            canvas_size: Size of the canvas (square).
            device: Torch device for computations.
        """
        self.num_strokes = num_strokes
        self.canvas_size = canvas_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # To be set by subclasses
        self.strokes_initialized = False

    @abstractmethod
    def init_strokes(
        self,
        target_image: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """
        Initialize stroke parameters.
        
        Args:
            target_image: Optional target image for attention-based initialization.
            **kwargs: Additional initialization parameters.
        """
        pass

    @abstractmethod
    def render(
        self,
        num_strokes: Optional[int] = None,
        return_alpha: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Render strokes to an image.
        
        Args:
            num_strokes: Number of strokes to render (for progressive rendering).
                         If None, render all strokes.
            return_alpha: Whether to return alpha channel separately.
            
        Returns:
            Rendered image tensor of shape (H, W, C) or (H, W, 3).
            If return_alpha is True, also returns alpha tensor of shape (H, W, 1).
        """
        pass

    @abstractmethod
    def get_points_parameters(self) -> List[torch.Tensor]:
        """
        Get stroke point parameters for optimization.
        
        Returns:
            List of tensors containing optimizable point parameters.
        """
        pass

    @abstractmethod
    def get_color_parameters(self) -> List[torch.Tensor]:
        """
        Get stroke color parameters for optimization.
        
        Returns:
            List of tensors containing optimizable color parameters.
        """
        pass

    @abstractmethod
    def get_width_parameters(self) -> List[torch.Tensor]:
        """
        Get stroke width parameters for optimization.
        
        Returns:
            List of tensors containing optimizable width parameters.
        """
        pass

    @abstractmethod
    def save_svg(self, path: Union[str, Path], num_strokes: Optional[int] = None) -> None:
        """
        Save strokes to SVG file.
        
        Args:
            path: Output path for SVG file.
            num_strokes: Number of strokes to save. If None, save all.
        """
        pass

    @abstractmethod
    def get_phase_stroke_count(self, phase_idx: int, phases: Union[List[Dict], List[PhaseConfig]]) -> int:
        """
        Get the number of strokes for a specific phase.
        
        Args:
            phase_idx: Index of the phase (0-based).
            phases: List of phase configurations (PhaseConfig objects or dicts).
            
        Returns:
            Number of strokes to render for this phase.
        """
        pass

    def get_all_parameters(self) -> List[torch.Tensor]:
        """
        Get all optimizable parameters.
        
        Returns:
            List of all parameter tensors.
        """
        params = []
        params.extend(self.get_points_parameters())
        params.extend(self.get_color_parameters())
        params.extend(self.get_width_parameters())
        return params

    def render_phase(
        self,
        phase_idx: int,
        phases: List[Dict],
        **render_kwargs,
    ) -> torch.Tensor:
        """
        Render strokes for a specific phase.
        
        Args:
            phase_idx: Index of the phase (0-based).
            phases: List of phase configurations.
            **render_kwargs: Additional rendering arguments.
            
        Returns:
            Rendered image for the phase.
        """
        num_strokes = self.get_phase_stroke_count(phase_idx, phases)
        return self.render(num_strokes=num_strokes, **render_kwargs)

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
        Compute overlay loss between different phases to minimize intersection.
        
        Args:
            phases: List of phase configurations.
            loss_type: Type of overlay loss ("dot", "iou", "dice", "hinge").
            blur_sigma: Sigma for Gaussian blur.
            blur_kernel_size: Kernel size for Gaussian blur.
            hinge_threshold: Threshold for hinge loss.
            save_debug_images: Whether to save debug images (used by NeuralPainter).
            debug_output_dir: Directory for debug images (used by NeuralPainter).
            **kwargs: Additional arguments (ignored by base implementation).
            
        Returns:
            Overlay loss value.
        """
        import torch.nn.functional as F
        from torchvision.transforms.functional import gaussian_blur
        
        assert len(phases) >= 2, "Overlay loss requires at least 2 phases"
        
        # Render each phase's strokes separately
        phase_masks = []
        prev_count = 0
        
        for phase_idx, phase in enumerate(phases):
            curr_count = self.get_phase_stroke_count(phase_idx, phases)
            
            if phase_idx == 0:
                # First phase: render strokes 0 to curr_count
                mask = self._render_mask(0, curr_count)
            else:
                # Subsequent phases: render delta strokes (prev_count to curr_count)
                mask = self._render_mask(prev_count, curr_count)
            
            phase_masks.append(mask)
            prev_count = curr_count
            
            if save_debug_images and debug_output_dir is not None:
                tensor_to_pil(mask, save_path=str(debug_output_dir / f"mask_{phase_idx}.png"))
        
        # Compute pairwise overlay loss between consecutive phases
        total_loss = torch.tensor(0.0, device=self.device)
        
        for i in range(len(phase_masks) - 1):
            mask_a = phase_masks[i]
            mask_b = phase_masks[i + 1]
            
            # Apply Gaussian blur for smooth gradients
            if blur_sigma > 0:
                mask_a = gaussian_blur(
                    mask_a.unsqueeze(0).unsqueeze(0),
                    kernel_size=blur_kernel_size,
                    sigma=blur_sigma
                ).squeeze()
                mask_b = gaussian_blur(
                    mask_b.unsqueeze(0).unsqueeze(0),
                    kernel_size=blur_kernel_size,
                    sigma=blur_sigma
                ).squeeze()
            
            if loss_type == "dot":
                # Dot product loss (penalize overlap)
                loss = torch.sum(mask_a * mask_b) / (self.canvas_size ** 2)
            elif loss_type == "iou":
                # Soft IoU loss
                intersection = torch.sum(mask_a * mask_b)
                union = torch.sum(mask_a) + torch.sum(mask_b) - intersection
                loss = intersection / (union + 1e-6)
            elif loss_type == "dice":
                # Soft Dice loss
                intersection = torch.sum(mask_a * mask_b)
                loss = (2 * intersection) / (torch.sum(mask_a) + torch.sum(mask_b) + 1e-6)
            elif loss_type == "hinge":
                # Hinge loss
                overlap = mask_a * mask_b
                loss = torch.mean(F.relu(overlap - hinge_threshold))
            else:
                loss = torch.tensor(0.0, device=self.device)
            
            total_loss = total_loss + loss
        
        return total_loss / max(len(phase_masks) - 1, 1)

    def _render_mask(self, start_idx: int, end_idx: int) -> torch.Tensor:
        """
        Render a binary mask for strokes from start_idx to end_idx.
        
        This is a helper method for overlay_loss. Subclasses may override
        for more efficient implementation.
        
        Args:
            start_idx: Starting stroke index (inclusive).
            end_idx: Ending stroke index (exclusive).
            
        Returns:
            Binary mask tensor of shape (H, W).
        """
        # Default implementation: render full image and extract alpha
        if end_idx <= start_idx:
            return torch.zeros(self.canvas_size, self.canvas_size, device=self.device)
        
        # This is a simplified version - subclasses should override for efficiency
        img = self.render(num_strokes=end_idx)
        if img.dim() == 3 and img.shape[-1] == 4:
            # Has alpha channel
            return img[:, :, 3]
        else:
            # Use grayscale as mask (inverted since strokes are usually dark on light)
            return 1.0 - img.mean(dim=-1)

    @abstractmethod
    def clamp_parameters(self) -> None:
        """
        Clamp parameters to valid ranges.
        
        This should be called after each optimization step to ensure
        parameters stay within valid bounds.
        """
        pass

    def to(self, device: torch.device) -> 'BasePainter':
        """
        Move painter to specified device.
        
        Args:
            device: Target device.
            
        Returns:
            Self for chaining.
        """
        self.device = device
        return self

    def state_dict(self) -> Dict[str, Any]:
        """
        Get state dictionary for checkpointing.
        
        Returns:
            Dictionary containing all state needed to restore the painter.
        """
        return {
            'num_strokes': self.num_strokes,
            'canvas_size': self.canvas_size,
            'strokes_initialized': self.strokes_initialized,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load state from dictionary.
        
        Args:
            state_dict: State dictionary from state_dict().
        """
        self.num_strokes = state_dict.get('num_strokes', self.num_strokes)
        self.canvas_size = state_dict.get('canvas_size', self.canvas_size)
        self.strokes_initialized = state_dict.get('strokes_initialized', False)


class PainterOptimizer:
    """
    Optimizer wrapper for painter parameters.
    
    Handles separate learning rates for different parameter groups
    (points, colors, widths) and learning rate scheduling.
    """

    def __init__(
        self,
        painter: BasePainter,
        lr_points: float = 0.8,
        lr_colors: float = 0.01,
        lr_widths: float = 0.1,
        optimizer_cls: type = torch.optim.Adam,
        **optimizer_kwargs,
    ):
        """
        Initialize optimizer for painter parameters.
        
        Args:
            painter: The painter instance to optimize.
            lr_points: Learning rate for point parameters.
            lr_colors: Learning rate for color parameters.
            lr_widths: Learning rate for width parameters.
            optimizer_cls: Optimizer class to use.
            **optimizer_kwargs: Additional optimizer arguments.
        """
        self.painter = painter
        
        param_groups = []
        
        # Points parameters
        points_params = painter.get_points_parameters()
        if points_params:
            param_groups.append({
                'params': points_params,
                'lr': lr_points,
                'name': 'points',
            })
        
        # Color parameters
        color_params = painter.get_color_parameters()
        if color_params:
            param_groups.append({
                'params': color_params,
                'lr': lr_colors,
                'name': 'colors',
            })
        
        # Width parameters
        width_params = painter.get_width_parameters()
        if width_params:
            param_groups.append({
                'params': width_params,
                'lr': lr_widths,
                'name': 'widths',
            })
        
        if not param_groups:
            raise ValueError("No parameters to optimize in painter")
        
        self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
        self.scheduler = None

    def zero_grad(self) -> None:
        """Zero gradients."""
        self.optimizer.zero_grad()

    def step(self) -> None:
        """Perform optimization step and clamp parameters."""
        self.optimizer.step()
        self.painter.clamp_parameters()
        
        if self.scheduler is not None:
            self.scheduler.step()

    def set_scheduler(
        self,
        scheduler_type: str,
        num_steps: int,
        **scheduler_kwargs,
    ) -> None:
        """
        Set learning rate scheduler.
        
        Args:
            scheduler_type: Type of scheduler ("cosine", "step", "linear").
            num_steps: Total number of optimization steps.
            **scheduler_kwargs: Additional scheduler arguments.
        """
        if scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_steps,
                **scheduler_kwargs,
            )
        elif scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                **scheduler_kwargs,
            )
        elif scheduler_type == "linear":
            def linear_lambda(step):
                return 1.0 - step / num_steps
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=linear_lambda,
            )
        else:
            self.scheduler = None

    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state dictionary."""
        state = {'optimizer': self.optimizer.state_dict()}
        if self.scheduler is not None:
            state['scheduler'] = self.scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load optimizer state from dictionary."""
        self.optimizer.load_state_dict(state_dict['optimizer'])
        if self.scheduler is not None and 'scheduler' in state_dict:
            self.scheduler.load_state_dict(state_dict['scheduler'])

