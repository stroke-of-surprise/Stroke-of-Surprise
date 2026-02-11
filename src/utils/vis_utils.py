import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from PIL import Image
from pathlib import Path
from typing import Optional, List, Union, Dict


def tensor2im(var: Tensor) -> Image.Image:
    """Convert tensor to PIL image.
    
    Args:
        var: Tensor of shape (C, H, W) or (1, C, H, W) or (H, W, C)
        
    Returns:
        PIL Image
    """
    var = var.clone()
    var = var.cpu().detach()
    
    # Handle different input shapes
    if var.dim() == 4:
        # (1, C, H, W) -> (C, H, W)
        var = var.squeeze(0)
    elif var.dim() == 3 and var.shape[0] == 1:
        # (1, H, W) -> (H, W) - grayscale, but we need to handle it
        var = var.squeeze(0)
    
    # Convert (C, H, W) to (H, W, C)
    if var.dim() == 3:
        var = var.permute(1, 2, 0)
    elif var.dim() == 2:
        # Grayscale image, add channel dimension
        var = var.unsqueeze(-1)
    
    var = var.numpy()
    var[var < 0] = 0
    var[var > 1] = 1
    var *= 255
    
    # Ensure we have the right shape for PIL
    if var.shape[2] == 1:
        # Grayscale
        var = var.squeeze(2)
    elif var.shape[2] != 3:
        raise ValueError(f"Unexpected number of channels: {var.shape[2]}")
    
    return Image.fromarray(var.astype("uint8"))


def convert_image_to_display(img) -> np.ndarray:
    """Convert various image formats to numpy array suitable for matplotlib display.
    
    Args:
        img: Input image (Tensor, PIL.Image, or np.ndarray)
        
    Returns:
        np.ndarray: Image in [0, 255] uint8 format with shape (H, W, C) or (H, W)
    """
    if isinstance(img, torch.Tensor):
        img_np = img.cpu().detach().numpy()
        if img_np.ndim == 4:  # (1, C, H, W) -> (C, H, W)
            img_np = img_np.squeeze(0)
        if img_np.ndim == 3 and img_np.shape[0] in [1, 3, 4]:  # CHW format
            img_np = np.transpose(img_np, (1, 2, 0))
    elif isinstance(img, Image.Image):
        img_np = np.array(img)
    else:
        img_np = np.array(img)
    
    # Normalize to [0, 255] uint8
    if img_np.dtype == np.float32 or img_np.dtype == np.float64:
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        else:
            img_np = img_np.clip(0, 255).astype(np.uint8)
    elif img_np.dtype != np.uint8:
        img_np = img_np.clip(0, 255).astype(np.uint8)
    
    return img_np


def normalize_attention_map(attn_map: np.ndarray) -> np.ndarray:
    """Normalize attention map to [0, 1] range.
    
    Args:
        attn_map: Attention map
        
    Returns:
        Normalized attention map
    """
    if attn_map.max() > attn_map.min():
        return (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
    return attn_map


def plot_attention_map(
    *,
    image: np.ndarray,
    attn: np.ndarray,
    threshold_map_list: List[np.ndarray],
    bg_threshold_map_list: List[np.ndarray],
    inds: np.ndarray,
    bg_inds: np.ndarray,
    output_path: Union[str, Path],
    title: str = "Attention Map Visualizations",
) -> None:
    """Plot attention maps with thresholds.
    
    Creates an 8-panel visualization showing:
    - Row 1: Target image, Attention map
    - Row 2-4: Threshold maps at different tau values (object and background)
    
    Args:
        image: Target image as numpy array
        attn: Attention map as numpy array
        threshold_map_list: List of threshold maps for object (different tau values)
        bg_threshold_map_list: List of threshold maps for background
        inds: Sampled object indices (shape: N, 2) in [h, w] format
        bg_inds: Sampled background indices (shape: N, 2) in [h, w] format
        output_path: Path to save the figure
        title: Figure title
    """
    num_rows_for_tau = 3
    fig, axes = plt.subplots(
        nrows=(1 + num_rows_for_tau), ncols=2, figsize=(6, 3 * (1 + num_rows_for_tau))
    )

    # Plot the images in the grid
    for i, ax in enumerate(axes.flat):

        if i == 0:
            # Plot image
            ax.imshow(image, interpolation="nearest")
            ax.axis("off")
            ax.set_title("Target Image")

        elif i == 1:
            # Plot Attention map
            ax.imshow(attn, interpolation="nearest")
            ax.axis("off")
            ax.set_title("Attention Map")

        else:
            if i in [2, 3]:
                # Take first map
                map_idx = 0
            elif i in [4, 5]:
                # Take middle map
                map_idx = len(threshold_map_list) // 2
            elif i in [6, 7]:
                # Take last map
                map_idx = -1
            else:
                raise ValueError(f"Doesn't support {num_rows_for_tau=} > 4")

            # Plot TAU attentions
            if i % 2 == 0:
                threshold_map = threshold_map_list[map_idx]
                # Plot Object attention
                threshold_map_ = normalize_attention_map(threshold_map)
                ax.imshow(threshold_map_, interpolation="nearest")
                ax.axis("off")
                ax.set_title(f"Object Softmax (tau idx={map_idx})")
                ax.scatter(
                    inds[:, 1], inds[:, 0], s=10, c="red", marker="o"
                )  # Add scatter points
            else:
                bg_threshold_map = bg_threshold_map_list[map_idx]
                # Plot Background attention
                bg_threshold_map_ = normalize_attention_map(bg_threshold_map)
                ax.imshow(bg_threshold_map_, interpolation="nearest")
                ax.axis("off")
                ax.set_title(f"Background Softmax (tau idx={map_idx})")
                ax.scatter(
                    bg_inds[:, 1], bg_inds[:, 0], s=10, c="red", marker="o"
                )  # Add scatter points

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_attention_map_dual_phase(
    *,
    image_A: np.ndarray,
    image_B: np.ndarray,
    attn_A: np.ndarray,
    attn_B: np.ndarray,
    threshold_map_list_A: List[np.ndarray],
    threshold_map_list_B: List[np.ndarray],
    bg_threshold_map_list_A: List[np.ndarray],
    bg_threshold_map_list_B: List[np.ndarray],
    inds_A: np.ndarray,
    inds_B: np.ndarray,
    bg_inds_A: np.ndarray,
    bg_inds_B: np.ndarray,
    output_path: Union[str, Path],
    title: str = "Attention Map Visualizations (Phase A & B)",
) -> None:
    """Plot attention maps for both phases (A and B) with thresholds.
    
    Creates a comprehensive visualization showing attention maps for both phases.
    
    Args:
        image_A, image_B: Target images for phase A and B
        attn_A, attn_B: Attention maps for phase A and B
        threshold_map_list_A, threshold_map_list_B: Threshold maps for object
        bg_threshold_map_list_A, bg_threshold_map_list_B: Threshold maps for background
        inds_A, inds_B: Sampled object indices in [h, w] format
        bg_inds_A, bg_inds_B: Sampled background indices
        output_path: Path to save the figure
        title: Figure title
    """
    num_rows_for_tau = 3
    # 2 columns for A and B, each has (1 + num_rows_for_tau) * 2 panels
    fig, axes = plt.subplots(
        nrows=(1 + num_rows_for_tau) * 2, ncols=2, 
        figsize=(8, 4 * (1 + num_rows_for_tau))
    )

    # --- Phase A (top half) ---
    for i in range(8):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        if i == 0:
            ax.imshow(image_A, interpolation="nearest")
            ax.axis("off")
            ax.set_title("Phase A: Target Image")
        elif i == 1:
            ax.imshow(attn_A, interpolation="nearest")
            ax.axis("off")
            ax.set_title("Phase A: Attention Map")
        else:
            if i in [2, 3]:
                map_idx = 0
            elif i in [4, 5]:
                map_idx = len(threshold_map_list_A) // 2
            else:
                map_idx = -1
            
            if i % 2 == 0:
                threshold_map = threshold_map_list_A[map_idx]
                threshold_map_ = normalize_attention_map(threshold_map)
                ax.imshow(threshold_map_, interpolation="nearest")
                ax.axis("off")
                ax.set_title(f"Phase A: Object (tau idx={map_idx})")
                ax.scatter(inds_A[:, 1], inds_A[:, 0], s=10, c="red", marker="o")
            else:
                bg_threshold_map = bg_threshold_map_list_A[map_idx]
                bg_threshold_map_ = normalize_attention_map(bg_threshold_map)
                ax.imshow(bg_threshold_map_, interpolation="nearest")
                ax.axis("off")
                ax.set_title(f"Phase A: Background (tau idx={map_idx})")
                ax.scatter(bg_inds_A[:, 1], bg_inds_A[:, 0], s=10, c="red", marker="o")

    # --- Phase B (bottom half) ---
    for i in range(8):
        row = (1 + num_rows_for_tau) + i // 2
        col = i % 2
        ax = axes[row, col]
        
        if i == 0:
            ax.imshow(image_B, interpolation="nearest")
            ax.axis("off")
            ax.set_title("Phase B: Target Image")
        elif i == 1:
            ax.imshow(attn_B, interpolation="nearest")
            ax.axis("off")
            ax.set_title("Phase B: Attention Map")
        else:
            if i in [2, 3]:
                map_idx = 0
            elif i in [4, 5]:
                map_idx = len(threshold_map_list_B) // 2
            else:
                map_idx = -1
            
            if i % 2 == 0:
                threshold_map = threshold_map_list_B[map_idx]
                threshold_map_ = normalize_attention_map(threshold_map)
                ax.imshow(threshold_map_, interpolation="nearest")
                ax.axis("off")
                ax.set_title(f"Phase B: Object (tau idx={map_idx})")
                ax.scatter(inds_B[:, 1], inds_B[:, 0], s=10, c="red", marker="o")
            else:
                bg_threshold_map = bg_threshold_map_list_B[map_idx]
                bg_threshold_map_ = normalize_attention_map(bg_threshold_map)
                ax.imshow(bg_threshold_map_, interpolation="nearest")
                ax.axis("off")
                ax.set_title(f"Phase B: Background (tau idx={map_idx})")
                ax.scatter(bg_inds_B[:, 1], bg_inds_B[:, 0], s=10, c="red", marker="o")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def simply_plot_image(
    image: Tensor,
    step: int,
    output_dir: Union[str, Path],
    title: str,
    caption: str,
) -> None:
    """Plot single image.
    
    Args:
        image: Image tensor
        step: Training step
        output_dir: Output directory
        title: Filename (without extension)
        caption: Image caption/title
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(tensor2im(image), interpolation="nearest")
    plt.axis("off")
    plt.title(caption)

    plt.savefig(f"{output_dir}/{title}")
    plt.close()


def plot_training_progress(
    outputs: List[Tensor],
    step: int,
    output_dir: Union[str, Path],
    title: str,
    caption: str,
    phase_labels: Optional[List[str]] = None,
) -> None:
    """Plot training progress with multiple outputs.
    
    Args:
        outputs: List of output tensors
        step: Training step
        output_dir: Output directory
        title: Filename
        caption: Figure title
        phase_labels: Optional labels for each output
    """
    n_outputs = len(outputs)
    if phase_labels is None:
        phase_labels = [f"Output {i+1}" for i in range(n_outputs)]
    
    fig, axes = plt.subplots(1, n_outputs, figsize=(5 * n_outputs, 5))
    if n_outputs == 1:
        axes = [axes]
    
    for idx, (output, label) in enumerate(zip(outputs, phase_labels)):
        axes[idx].imshow(tensor2im(output), interpolation="nearest")
        axes[idx].axis("off")
        axes[idx].set_title(label)
    
    fig.suptitle(f"{caption} (Step {step})", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{title}")
    plt.close(fig)


def save_attention_maps(
    phase_attn_maps: List[np.ndarray],
    phase_images: List,
    phase_attn_data: List[Dict],
    phase_names: List[str],
    output_dir: Union[str, Path],
) -> None:
    """Save attention maps visualization for all phases using vis_utils.
    
    Creates visualizations showing:
    - Target images
    - Attention maps
    - Threshold maps at different tau values (object and background)
    - Sampled stroke positions
    
    This matches the coach.py visualization format for consistency.
    
    Args:
        phase_attn_maps: List of attention maps for each phase
        phase_images: List of input images for each phase (PIL Image or numpy array)
        phase_attn_data: List of dictionaries containing attention data for each phase:
            - 'attn_map_soft_list': List of threshold maps for object
            - 'bg_attn_map_soft_list': List of threshold maps for background
            - 'inds': Sampled object indices
            - 'bg_inds': Sampled background indices
        phase_names: List of phase names (e.g., ['phase_1', 'phase_2', ...])
        output_dir: Output directory where attention maps will be saved
    """
    # Save attention maps for each phase
    for phase_idx, (attn_map, image, attn_data, phase_name) in enumerate(
        zip(phase_attn_maps, phase_images, phase_attn_data, phase_names)
    ):
        # Convert image to display format
        img_np = convert_image_to_display(image)
        
        # Convert attention map to numpy if needed
        if isinstance(attn_map, torch.Tensor):
            attn_map = attn_map.cpu().detach().numpy()
        
        # Normalize attention map
        attn_norm = normalize_attention_map(attn_map)
        
        # Save attention map visualization for this phase
        plot_attention_map(
            image=img_np,
            attn=attn_norm,
            threshold_map_list=attn_data['attn_map_soft_list'],
            bg_threshold_map_list=attn_data['bg_attn_map_soft_list'],
            inds=attn_data['inds'],
            bg_inds=attn_data['bg_inds'],
            output_path=output_dir / f"attention_map_{phase_name}.png",
            title=f"{phase_name}: Attention Map Visualizations",
        )
        
        # Also save attention map as numpy file for later analysis
        np.save(output_dir / f"attn_map_{phase_name}.npy", attn_map)
    
    print(f"Attention maps saved to {output_dir}")

