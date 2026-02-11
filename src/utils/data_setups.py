"""Image preprocessing utilities."""

import logging
from pathlib import Path
from typing import Tuple, Optional, Union

import torch
from PIL import Image
from torchvision import transforms

from ..configs.data_config import DataConfig

logger = logging.getLogger(__name__)

# Type aliases
ImageType = Union[Image.Image, torch.Tensor]
MaskType = Optional[Union[Image.Image, torch.Tensor]]


def create_transform(size: int) -> transforms.Compose:
    """Create transform pipeline for image preprocessing."""
    return transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
        ]
    )


def segment_image(
    image_path: Path, transform: transforms.Compose, target_size: int
) -> Tuple[Image.Image, torch.Tensor, torch.Tensor]:
    """Segment foreground object from image."""
    try:
        # Initialize segmentation pipeline
        from transformers import pipeline as hf_pipeline
        pipe = hf_pipeline(
            "image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True
        )

        # Get segmented image
        image_tmp = pipe(str(image_path))
        image_tmp = image_tmp.resize((target_size, target_size))

        # Create white background image and paste segmented content
        image = Image.new("RGB", (target_size, target_size), (255, 255, 255))
        image.paste(image_tmp, mask=image_tmp.split()[3])

        # Get mask
        mask = pipe(str(image_path), return_mask=True)
        mask = transform(mask)
        mask = (mask >= 0.5).float()  # Binarize mask

        # Convert image to tensor
        image_tensor = transform(image).unsqueeze(0)

        return image, mask, image_tensor

    except Exception as e:
        logger.error(f"Failed to segment image {image_path}: {str(e)}")
        raise RuntimeError(f"Image segmentation failed: {str(e)}")


def load_regular_image(
    image_path: Path, transform: transforms.Compose
) -> Tuple[Image.Image, None, torch.Tensor]:
    """Load image without segmentation.

    Args:
        image_path: Input image path
        transform: Transform pipeline

    Returns:
        tuple: (image, None, image_tensor)
    """
    try:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)
        return image, None, image_tensor

    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {str(e)}")
        raise RuntimeError(f"Image loading failed: {str(e)}")


def prepare_image(
    image_path: Path, cfg: DataConfig
) -> Tuple[ImageType, MaskType, torch.Tensor]:
    """Prepare image for training/inference.

    Args:
        image_path: Input image path
        cfg: Data configuration

    Returns:
        tuple: (processed_image, mask, image_tensor)
    """
    logger.info(f"Preparing image: {image_path}")

    if not isinstance(image_path, Path):
        image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    transform = create_transform(cfg.render_size)

    try:
        # Check if segment_object is configured
        segment_object = getattr(cfg, 'segment_object', False)
        if segment_object:
            logger.debug("Applying foreground segmentation")
            return segment_image(image_path, transform, cfg.render_size)
        else:
            logger.debug("Loading image without segmentation")
            return load_regular_image(image_path, transform)

    except Exception as e:
        logger.error(f"Failed to prepare image {image_path}: {str(e)}")
        raise

