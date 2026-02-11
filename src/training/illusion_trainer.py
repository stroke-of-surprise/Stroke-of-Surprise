"""
Illusion Trainer - Core training logic for multi-phase SDS optimization.

This module implements the "Stroke of Surprise" training approach where
different phases of strokes are optimized against different text prompts,
creating illusion sketches that transform as strokes are added.
"""

import os
import gc
import logging
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from ..configs import TrainConfig
from ..painters import BasePainter, PainterOptimizer, BezierPainter, BsplinePainter, NeuralPainter, NeuralPainterOptimizer
from ..diffusion import SDSLoss, build_precompute_prompts
from ..utils import read_svg, create_phase_grid_with_diffs, tensor_to_pil, svg_to_pil
from ..utils.log_utils import save_config

logger = logging.getLogger(__name__)


class IllusionTrainer:
    """
    Trainer for illusion sketch generation using multi-phase SDS optimization.
    
    This trainer supports:
    - Multiple painter types (Bezier, B-spline, Neural)
    - Multi-phase training with different prompts per phase
    - Overlay loss to minimize intersection between phases
    - Flexible SDS loss with optional LoRA and precomputed embeddings
    """

    def __init__(self, cfg: TrainConfig, device: torch.device = None):
        """
        Initialize the trainer.
        
        Args:
            cfg: Training configuration.
            device: Torch device for computation.
        """
        self.cfg = cfg
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Setup output directories
        self._setup_directories()

        # Initialize painter
        self.painter = self._create_painter()
        
        # Initialize SDS losses for each phase
        self.phase_sds_losses = []
        
        # Training state
        self.current_epoch = 0
        self.optimizer = None
        
        # Pretrain state (for Neural painter)
        self.points_init = []  # Target points for each phase

    def _setup_directories(self) -> None:
        """Setup output directories."""
        self.cfg.log.exp_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.log.svg_logs_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.log.png_logs_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.log.setup_dir.mkdir(parents=True, exist_ok=True)
        
        if self.cfg.optim.pretrain:
            self.cfg.log.pretrain_logs_dir.mkdir(parents=True, exist_ok=True)

    def _create_painter(self) -> BasePainter:
        """Create painter based on configuration."""
        painter_type = self.cfg.painter_type
        data_cfg = self.cfg.data
        
        if painter_type == "bezier":
            bezier_cfg = self.cfg.bezier
            painter = BezierPainter(
                num_strokes=data_cfg.num_strokes,
                canvas_size=data_cfg.render_size,
                device=self.device,
                num_segments=bezier_cfg.num_segments,
                control_points_per_seg=bezier_cfg.control_points_per_seg,
                width=bezier_cfg.width,
                optimize_color=data_cfg.optimize_color,
                background_image=self._load_background_image(),
                init_strokes_svg=data_cfg.init_strokes_svg,
            )
        elif painter_type == "bspline":
            bspline_cfg = self.cfg.bspline
            painter = BsplinePainter(
                num_strokes=data_cfg.num_strokes,
                canvas_size=data_cfg.render_size,
                device=self.device,
                spline_degree=bspline_cfg.spline_degree,
                multiplicity=bspline_cfg.multiplicity,
                num_control_points=bspline_cfg.num_control_points,
                width=bspline_cfg.width,
                min_width=bspline_cfg.min_width,
                max_width=bspline_cfg.max_width,
                optimize_width=bspline_cfg.width_optim,
                init_mode=bspline_cfg.init_mode,
                init_strokes_svg=data_cfg.init_strokes_svg,
            )
        elif painter_type == "neural":
            neural_cfg = self.cfg.neural
            painter = NeuralPainter(
                num_strokes=data_cfg.num_strokes,
                canvas_size=data_cfg.render_size,
                device=self.device,
                num_segments=data_cfg.num_segments,
                control_points_per_seg=data_cfg.control_points_per_seg,
                width=data_cfg.width,
                is_closed=data_cfg.is_closed,
                radius=data_cfg.radius,
                # Pretrain config
                sd_model=neural_cfg.text2img_model,
                lora_weights=neural_cfg.lora_weights,
                # MLP config
                mlp_dim=neural_cfg.mlp_dim,
                mlp_num_layers=neural_cfg.mlp_num_layers,
                input_dim=neural_cfg.input_dim,
                use_nested_dropout=neural_cfg.use_nested_dropout,
                truncation_start_idx=neural_cfg.truncation_start_idx,
                points_prediction_scale=neural_cfg.points_prediction_scale,
                # Color config
                use_color=neural_cfg.use_color,
                toggle_color=neural_cfg.toggle_color,
                toggle_color_method=neural_cfg.toggle_color_method,
                toggle_color_input_dim=neural_cfg.toggle_color_input_dim,
                toggle_color_bg_colors=neural_cfg.toggle_color_bg_colors,
                toggle_color_init_eps=neural_cfg.toggle_color_init_eps,
                toggle_sample_random_color_prob=neural_cfg.toggle_sample_random_color_prob,
                # Dropout config
                use_dropout_value=neural_cfg.use_dropout_value,
                # Other
                use_background=getattr(data_cfg, 'use_background', True),
                regular_polygon_closed_shape_init=True,
            )
        else:
            raise ValueError(f"Unknown painter type: {painter_type}")
        
        return painter

    def _load_background_image(self) -> Optional[Image.Image]:
        """Load background image if specified."""
        bg_path = self.cfg.data.background_image
        if bg_path and os.path.isfile(bg_path):
            print(f"Loaded background image: {bg_path}")
            return Image.open(bg_path)
        return None

    def _init_sds_losses(self) -> None:
        """Initialize SDS loss for each phase."""
        self.phase_sds_losses = []
        
        optim_cfg = self.cfg.optim
        neural_cfg = self.cfg.neural if self.cfg.painter_type == "neural" else None
        
        # Get SD version from config
        sd_version = self.cfg.sd_version
        print(f"Using Stable Diffusion version: {sd_version}")
        
        # B-spline uses grayscale (rgb=False), others use RGB
        use_rgb = self.cfg.painter_type != "bspline"
        
        # Get grad_method and time_schedule from config
        grad_method = optim_cfg.sds_grad_method
        time_schedule = optim_cfg.sds_time_schedule
        loss_style = optim_cfg.sds_loss_style
        
        print(f"SDS settings: rgb={use_rgb}, grad_method={grad_method}, time_schedule={time_schedule}, loss_style={loss_style}")
        
        for phase_idx, phase in enumerate(self.cfg.phases):
            caption = phase.caption
            
            # Add prompt suffix
            full_caption = f"{caption} {self.cfg.data.text_prompt_suffix}".strip()
            
            print(f"  Phase {phase_idx + 1}: caption='{caption}', weight={phase.weight}")    
            # Create SDS loss
            sds_loss = SDSLoss(
                full_caption,
                version=sd_version,
                augment=0,
                rgb=use_rgb,
                seed=self.cfg.seed,
                t_range=[optim_cfg.sds_t_range_min, optim_cfg.sds_t_range_max],
                guidance_scale=optim_cfg.sds_guidance_scale,
                grad_method=grad_method,
                time_schedule=time_schedule,
                lora_weights=neural_cfg.lora_weights if neural_cfg else None,
                loss_style=loss_style,
            )
            
            self.phase_sds_losses.append(sds_loss)

    def _init_optimizer(self) -> Union[PainterOptimizer, NeuralPainterOptimizer]:
        """Initialize optimizer for painter parameters."""
        optim_cfg = self.cfg.optim
        
        # For Neural painter, use NeuralPainterOptimizer (AdamW with NeuralSVG settings)
        if self.cfg.painter_type == "neural":
            optimizer = NeuralPainterOptimizer(
                self.painter,
                lr=optim_cfg.learning_rate,
                lr_pretrain=optim_cfg.learning_rate_pretrain,
                weight_decay=getattr(optim_cfg, 'weight_decay', 0.0),
                pretrain=False,  # Main training, not pretrain
            )
            
            optimizer.set_scheduler(
                optim_cfg.scheduler_type.value,
                max_steps=self.cfg.num_iter,
                warmup_steps=optim_cfg.warmup_steps,
                lr_final=optim_cfg.target_lr,
            )
            
            print(f"Using NeuralPainterOptimizer: lr={optim_cfg.learning_rate}, scheduler={optim_cfg.scheduler_type}")
            return optimizer
        
        # For other painters, use PainterOptimizer
        bspline_cfg = self.cfg.bspline if self.cfg.painter_type == "bspline" else None
        
        # Use bspline-specific learning rates if available
        if bspline_cfg:
            lr_points = bspline_cfg.lr_pos
            lr_widths = bspline_cfg.lr_width
            lr_colors = optim_cfg.learning_rate_color
            print(f"Using B-spline learning rates: lr_pos={lr_points}, lr_width={lr_widths}")
        else:
            lr_points = optim_cfg.learning_rate
            lr_widths = optim_cfg.learning_rate
            lr_colors = optim_cfg.learning_rate_color
        
        optimizer = PainterOptimizer(
            self.painter,
            lr_points=lr_points,
            lr_colors=lr_colors,
            lr_widths=lr_widths,
        )
        
        # Set up scheduler
        if optim_cfg.scheduler_type != "constant":
            optimizer.set_scheduler(
                optim_cfg.scheduler_type,
                self.cfg.num_iter,
            )
        
        return optimizer

    def train(self) -> None:
        """Run the main training loop."""
        optim_cfg = self.cfg.optim
        
        # For Neural painter: handle pretrain phase
        if self.cfg.painter_type == "neural" and optim_cfg.pretrain:
            print("Running pretrain phase for Neural painter...")
            self.painter.pretrain(
                phases=self.cfg.phases,
                data_cfg=self.cfg.data,
                optim_cfg=optim_cfg,
                log_cfg=self.cfg.log,
                device=str(self.device),
                generate_target_image=self.cfg.data.generate_target_image,
            )
            print("Pretrain completed, switching to SDS training...")
        
        # Initialize painter strokes (for non-neural or after pretrain)
        if self.cfg.painter_type != "neural" or not optim_cfg.pretrain:
            print("Initializing painter strokes...")
            self.painter.init_strokes(phases=self.cfg.phases)
        
        # Save initial SVG
        self.painter.save_svg(
            self.cfg.log.exp_dir / "setup" / "init_svg",
            save_groups=True,
            phases=self.cfg.phases,
        )
        
        # Initialize SDS losses
        print("Initializing SDS losses...")
        self._init_sds_losses()
        
        # Initialize optimizer
        self.optimizer = self._init_optimizer()
        
        # Training loop
        print("Starting optimization...")
        self._training_loop()
        
        # Save final results
        self._save_final_results()
        
        # Cleanup
        self._cleanup()

    def _training_loop(self) -> None:
        """Main training loop."""
        num_iter = self.cfg.num_iter
        save_interval = self.cfg.log.save_interval
        optim_cfg = self.cfg.optim
        
        epoch_range = tqdm(range(num_iter + 1), desc="Training")
        
        for epoch in epoch_range:
            self.current_epoch = epoch
            self.optimizer.zero_grad()
            
            # Compute multi-phase SDS loss
            total_loss = torch.tensor(0.0, device=self.device)
            
            # Get toggle color for this step (for training visualization)
            toggle_color = None
            if self.cfg.painter_type == "neural" and hasattr(self.cfg.neural, 'toggle_color'):
                if self.cfg.neural.toggle_color and self.cfg.neural.toggle_color_bg_colors:
                    import random
                    toggle_color = random.choice(self.cfg.neural.toggle_color_bg_colors)
            
            for phase_idx, phase in enumerate(self.cfg.phases):
                # Get number of strokes for this phase
                n_strokes = self.painter.get_phase_stroke_count(phase_idx, self.cfg.phases)
                
                # Render image (with toggle color for Neural painter)
                if self.cfg.painter_type == "neural":
                    img = self.painter.get_image(
                        truncation_indices=[n_strokes],
                        # num_strokes=n_strokes,
                        toggle_color_value=toggle_color,
                    )
                else:
                    img = self.painter.get_image(num_strokes=n_strokes)
                
                # For bspline, use grayscale input; for others, use RGB
                if self.cfg.painter_type == "bspline":
                    # Convert RGB to grayscale (H, W)
                    img_for_sds = img[0].mean(dim=0)
                elif self.cfg.painter_type == "neural":
                    img_for_sds = img
                else:
                    # Convert from (1, 3, H, W) to (H, W, 3)
                    img_for_sds = img[0].permute(1, 2, 0)
                
                # Compute SDS loss
                # For Neural painter with toggle_color, add background color to prompt
                # This follows NeuralSVG's approach where the prompt includes "on {color} background"
                sds_loss = self.phase_sds_losses[phase_idx]
                custom_text = None
                if self.cfg.painter_type == "neural" and toggle_color is not None:
                    # Modify prompt to include background color (NeuralSVG style)
                    base_caption = phase.caption
                    custom_text = f"{base_caption} isolated on {toggle_color} background"
                phase_loss = sds_loss(
                    img_for_sds, 
                    step=epoch, 
                    num_steps=num_iter,
                    grad_scale=optim_cfg.sds_grad_scale,
                    custom_text=custom_text,
                )
                
                weight = phase.weight
                total_loss = total_loss + weight * phase_loss
                
                # Save per-phase images at intervals
                if epoch % save_interval == 0:
                    group_dir = self.cfg.log.exp_dir / "group_images"
                    img_pil = tensor_to_pil(img, save_path=group_dir / f"iter_{epoch:04d}_{n_strokes}s.png")
                    img_pil.save(group_dir / f"final_{n_strokes}s.png")

            # Add overlay loss
            overlay_loss = self._compute_overlay_loss()
            if overlay_loss is not None:
                total_loss = total_loss + overlay_loss
            
            # Add B-spline smoothing loss
            if self.cfg.painter_type == "bspline":
                smoothing_loss = self._compute_smoothing_loss()
                total_loss = total_loss + smoothing_loss
            
            # Backward and optimize
            total_loss.backward()
            
            # Clip gradients to prevent numerical instability (especially for Neural painter)
            if self.cfg.optim.use_clip_grad:
                if self.cfg.painter_type == "neural":
                    # Neural painter: separate clip for points and colors MLP
                    self._clip_neural_gradients()
                else:
                    # Other painters: unified clip for all params
                    all_params = []
                    for param_group in self.optimizer.optimizer.param_groups:
                        all_params.extend(param_group['params'])
                    if all_params:
                        torch.nn.utils.clip_grad_norm_(
                            all_params, 
                            max_norm=self.cfg.optim.clip_grad_max_norm
                        )
            
            self.optimizer.step()
            
            # Clamp parameters to valid ranges (especially for B-spline widths)
            self.painter.clamp_parameters()
            
            # Save logs
            if epoch % save_interval == 0:
                self.log_all_phases(epoch)
                self.log_colorful_bg(epoch)
            
            # Update progress bar
            epoch_range.set_postfix(loss=f"{total_loss.item():.4f}")

    def _clip_neural_gradients(self) -> None:
        """Apply gradient clipping for Neural painter (separate points/colors).
        
        This follows NeuralSVG coach.py's approach:
        - Separate clipping for mlp_points and mlp_color with different max_norms
        - Only clips parameters with requires_grad=True and non-None gradients
        """
        optim_cfg = self.cfg.optim
        mlp = self.painter.mlp
        
        if mlp is None:
            return
        
        # Clip points MLP gradients
        if hasattr(mlp, "mlp_points"):
            params_points = list(
                filter(
                    lambda p: p.requires_grad and p.grad is not None,
                    mlp.mlp_points.parameters(),
                )
            )
            if params_points:
                torch.nn.utils.clip_grad_norm_(
                    params_points,
                    max_norm=optim_cfg.clip_grad_max_norm_points,
                )
        
        # Clip colors MLP gradients
        if hasattr(mlp, "mlp_color"):
            params_colors = list(
                filter(
                    lambda p: p.requires_grad and p.grad is not None,
                    mlp.mlp_color.parameters(),
                )
            )
            if params_colors:
                torch.nn.utils.clip_grad_norm_(
                    params_colors,
                    max_norm=optim_cfg.clip_grad_max_norm_colors,
                )

    # =============================== Losses ===============================
    def _compute_overlay_loss(self) -> Optional[torch.Tensor]:
        """Compute overlay loss between phases."""
        optim_cfg = self.cfg.optim
        
        if optim_cfg.overlay_loss_type == "None" or optim_cfg.overlay_loss_weight <= 0:
            return None
        assert len(self.cfg.phases) >= 2, "Overlay loss requires at least 2 phases"
        
        # Determine whether to save debug images
        save_debug = (
            self.current_epoch % self.cfg.log.save_interval == 0 and
            self.cfg.painter_type == "neural"
        )
        debug_dir = self.cfg.log.exp_dir / "overlay" if save_debug else None
        
        overlay_loss = self.painter.overlay_loss(
            self.cfg.phases,
            loss_type=optim_cfg.overlay_loss_type,
            blur_sigma=getattr(optim_cfg, 'overlay_blur_sigma', 0),
            blur_kernel_size=getattr(optim_cfg, 'overlay_blur_kernel_size', 0),
            hinge_threshold=getattr(optim_cfg, 'overlay_hinge_threshold', 0.0),
            save_debug_images=save_debug,
            debug_output_dir=debug_dir,
        )
        
        return optim_cfg.overlay_loss_weight * overlay_loss

    def _compute_smoothing_loss(self) -> torch.Tensor:
        """Compute B-spline smoothing loss."""
        if not isinstance(self.painter, BsplinePainter):
            return torch.tensor(0.0, device=self.device)
        
        bspline_cfg = self.cfg.bspline
        return self.painter.compute_smoothing_loss(
            deriv_order=bspline_cfg.smoothing_deriv,
            weight=bspline_cfg.smoothing_weight,
        )

    # =============================== Logging ===============================
    def log_all_phases(self, epoch: int) -> None:
        """Save training checkpoint."""
        svg_dir = self.cfg.log.svg_logs_dir
        png_dir = self.cfg.log.png_logs_dir
        
        # Save SVG
        svg_files = self.painter.save_svg(svg_dir / f"iter_{epoch:04d}", save_groups=True, phases=self.cfg.phases)
        
        img_pils, img_paths = svg_to_pil(svg_files, device=self.device, save_dir=png_dir)
        for img_pil, img_path in zip(img_pils, img_paths):
            img_pil.save(img_path.replace(f'iter_{epoch:04d}', 'final'))

    def log_colorful_bg(self, epoch: int) -> None:
        """Save colorful background images during training with toggle colors (for Neural painter).
        
        Creates a grid showing all phases (rows) and all toggle colors (columns).
        """
        if self.cfg.painter_type != "neural":
            return
        
        logs_dir = self.cfg.log.exp_dir / "toggle_color_logs"
        os.makedirs(logs_dir, exist_ok=True)
        
        # Get toggle colors
        toggle_colors = ["white"]
        if hasattr(self.cfg.neural, 'toggle_color'):
            if self.cfg.neural.toggle_color and self.cfg.neural.toggle_color_bg_colors:
                toggle_colors = self.cfg.neural.toggle_color_bg_colors
        
        num_phases = len(self.cfg.phases)
        num_colors = len(toggle_colors)
        
        # Store images: all_images[phase_idx][color_idx] = image_pil
        all_images = [[None for _ in range(num_colors)] for _ in range(num_phases)]
        
        self.painter.mlp.eval()
        with torch.no_grad():
            # Loop through all phases
            for phase_idx, phase in enumerate(self.cfg.phases):
                # Get number of strokes for this phase
                n_strokes = self.painter.get_phase_stroke_count(phase_idx, self.cfg.phases)
                
                # Loop through all toggle colors
                for color_idx, toggle_color in enumerate(toggle_colors):
                    # Render image for this phase and toggle color
                    img = self.painter.get_image(
                        truncation_indices=[n_strokes],
                        toggle_color_value=toggle_color,
                    )
                    
                    # Convert to PIL
                    img_np = img[0].permute(1, 2, 0).detach().cpu().numpy()
                    img_pil = Image.fromarray((img_np * 255).astype("uint8"), "RGB")
                    
                    all_images[phase_idx][color_idx] = img_pil
        
        self.painter.mlp.train()
        
        # Create grid image: rows = num_phases, cols = num_colors
        fig, axes = plt.subplots(num_phases, num_colors, figsize=(3 * num_colors, 3 * num_phases))
        
        # Handle single phase or single color case
        if num_phases == 1:
            axes = axes.reshape(1, -1)
        if num_colors == 1:
            axes = axes.reshape(-1, 1)
        
        # Plot images in grid
        for phase_idx, phase in enumerate(self.cfg.phases):
            n_strokes = self.painter.get_phase_stroke_count(phase_idx, self.cfg.phases)
            
            for color_idx, toggle_color in enumerate(toggle_colors):
                ax = axes[phase_idx, color_idx]
                img_pil = all_images[phase_idx][color_idx]
                
                ax.imshow(img_pil)
                ax.axis('off')
                
                # Set title: color name on top row
                if phase_idx == 0:
                    ax.set_title(f"{toggle_color}", fontsize=10)
                
                # Set ylabel: phase info on first column
                if color_idx == 0:
                    title = f"{phase.name}\n({n_strokes} strokes)"
                    ax.set_ylabel(title, fontsize=10, rotation=0, ha='right', va='center')
        
        fig.suptitle(f"Training Step {epoch}", fontsize=14)
        plt.tight_layout()
        
        plt.savefig(logs_dir / f"iter_{epoch}.jpg", dpi=100, bbox_inches='tight')
        plt.close(fig)



    def _save_final_results(self) -> None:
        """Save final results after training."""
        exp_dir = self.cfg.log.exp_dir
        
        # Save phase-wise sketches
        for phase_idx in range(len(self.cfg.phases)):
            n_strokes = self.painter.get_phase_stroke_count(phase_idx, self.cfg.phases)
            phase_name = f"sketch_p{phase_idx + 1}"  # Keep legacy naming for output files
            
            # Only last phase saves group-colored version
            save_groups = (phase_idx == len(self.cfg.phases) - 1)
            
            svg_files = self.painter.save_svg(
                exp_dir / phase_name,
                num_strokes=n_strokes,
                save_groups=save_groups,
                phases=self.cfg.phases,
            )
            
            for svg_file in svg_files:
                try:
                    sketch = read_svg(svg_file, self.device, multiply=True)
                    sketch_np = sketch.cpu().numpy()
                    sketch_pil = Image.fromarray((sketch_np * 255).astype('uint8'), 'RGB')
                    
                    # Handle group file renaming
                    if "_group" in svg_file:
                        new_svg = str(exp_dir / "sketch_group.svg")
                        os.rename(svg_file, new_svg)
                        svg_file = new_svg
                    
                    png_path = svg_file.replace('.svg', '.png')
                    sketch_pil.save(png_path)
                    print(f"Phase {phase_idx + 1} saved: {png_path}")
                except Exception as e:
                    print(f"Warning: Could not save {svg_file}: {e}")
        
        # Create phase grid
        try:
            create_phase_grid_with_diffs(str(exp_dir), self.cfg.phases)
        except Exception as e:
            print(f"Warning: Could not create phase grid: {e}")
        
        # Save stroke data for B-spline (preserves varying widths and stroke order)
        if self.cfg.painter_type == "bspline" and hasattr(self.painter, 'save_strokes'):
            try:
                strokes_path = exp_dir / "strokes.pkl"
                self.painter.save_strokes(strokes_path)
            except Exception as e:
                print(f"Warning: Could not save strokes: {e}")
        
        # Run inference for Neural painter
        if self.cfg.painter_type == "neural":
            self._run_inference()
        
        # Save config
        save_config(self.cfg, self.cfg.log.exp_dir)

    def _run_inference(self) -> None:
        """Run inference and save final images for all toggle colors and phases.
        
        Follows NeuralSVG coach.py inference logic:
        - Saves images for each toggle color
        - Saves both phase A and phase B versions for illusion sketches
        - Saves SVG files with toggle color suffixes
        """
        exp_dir = self.cfg.log.exp_dir
        inference_dir = exp_dir / "inference"
        inference_dir.mkdir(exist_ok=True)
        
        # Get toggle colors
        toggle_colors = [None]  # Default: no toggle
        if hasattr(self.cfg.neural, 'toggle_color'):
            if self.cfg.neural.toggle_color and self.cfg.neural.toggle_color_bg_colors:
                toggle_colors = self.cfg.neural.toggle_color_bg_colors
        
        # Get stroke counts for each phase (for backward compatibility)
        num_strokes_A = self.cfg.phases[0].num_strokes if self.cfg.phases else self.cfg.data.num_strokes // 2
        num_strokes_B = self.cfg.data.num_strokes
        
        # Set model to eval mode
        if hasattr(self.painter, 'mlp') and self.painter.mlp is not None:
            self.painter.mlp.eval()
        
        with torch.no_grad():
            for bg_color in toggle_colors:
                color_suffix = bg_color.replace(" ", "-") if bg_color else "no_toggle_bg"
                
                # Save phase A image
                try:
                    img_A = self.painter.get_image(
                        truncation_indices=[num_strokes_A],
                        toggle_color_value=bg_color,
                    )
                    img_A_np = img_A[0].permute(1, 2, 0).detach().cpu().numpy()
                    img_A_pil = Image.fromarray((img_A_np * 255).astype("uint8"), "RGB")
                    img_A_pil.save(inference_dir / f"{color_suffix}_p1.jpg")
                except Exception as e:
                    print(f"Warning: Could not save phase A inference image: {e}")
                
                # Save phase B image
                try:
                    img_B = self.painter.get_image(
                        truncation_indices=[num_strokes_B],
                        toggle_color_value=bg_color,
                    )
                    img_B_np = img_B[0].permute(1, 2, 0).detach().cpu().numpy()
                    img_B_pil = Image.fromarray((img_B_np * 255).astype("uint8"), "RGB")
                    img_B_pil.save(inference_dir / f"{color_suffix}_p2.jpg")
                except Exception as e:
                    print(f"Warning: Could not save phase B inference image: {e}")
                
                # Save SVG files
                try:
                    svg_logs_dir = exp_dir / "svg_logs"
                    self.painter.save_svg(
                        svg_logs_dir / f"final_svg_{color_suffix}_p1",
                        num_strokes=num_strokes_A,
                        toggle_color_value=bg_color,
                    )
                    self.painter.save_svg(
                        svg_logs_dir / f"final_svg_{color_suffix}_p2",
                        num_strokes=num_strokes_B,
                        toggle_color_value=bg_color,
                    )
                except Exception as e:
                    print(f"Warning: Could not save SVG for {color_suffix}: {e}")
        
        # Set model back to train mode
        if hasattr(self.painter, 'mlp') and self.painter.mlp is not None:
            self.painter.mlp.train()
        
        print(f"Inference results saved to {inference_dir}")

    def _cleanup(self) -> None:
        """Clean up resources."""
        del self.phase_sds_losses
        del self.optimizer
        gc.collect()
        torch.cuda.empty_cache()
        print("Cleanup complete")


def train_from_config(cfg: TrainConfig) -> None:
    """
    Train illusion sketch from configuration.
    
    Args:
        cfg: Training configuration.
    """
    # Set seed
    import random
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
    
    # Print configuration
    cfg.print_config()
    
    # Create and run trainer
    trainer = IllusionTrainer(cfg)
    trainer.train()

