"""
Attention map utilities for stroke initialization.

This module provides functions to:
1. Extract CLIP attention maps from images
2. Initialize stroke positions based on attention maps
"""

import torch
import numpy as np
import CLIP_.clip as clip

from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.transform import resize


def get_clip_attention_map(
    input_image: Image.Image, image_size: int, device: str = "cuda"
):
    """
    Extract CLIP attention map from an image.
    
    Args:
        input_image: PIL Image to process.
        image_size: Target size for the attention map.
        device: Torch device.
        
    Returns:
        Tuple of (preprocessed_image, attention_map)
    """
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    model.eval().to(device)
    input_image = preprocess(input_image).to(device)
    attn_map = get_attention_map(
        input_image, model, image_size=image_size, device=device
    )
    del model
    return input_image, attn_map


def get_attention_map(image, model, device, image_size: int) -> torch.Tensor:
    """
    Compute attention map from CLIP model.
    
    Args:
        image: Preprocessed image tensor.
        model: CLIP model.
        device: Torch device.
        image_size: Target size for the attention map.
        
    Returns:
        Attention map as numpy array.
    """
    images = image.repeat(1, 1, 1, 1)
    model.encode_image(images, mode="saliency")
    model.zero_grad()
    image_attn_blocks = list(
        dict(model.visual.transformer.resblocks.named_children()).values()
    )
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(
        num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype
    ).to(device)
    R = R.unsqueeze(0).expand(1, num_tokens, num_tokens)

    cams = []
    for blk in image_attn_blocks:
        cam = blk.attn_probs.detach()
        cam = cam.reshape(1, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0)
        cam = cam.clamp(min=0).mean(dim=1)
        cams.append(cam)
        R = R + torch.bmm(cam, R)

    cams_avg = torch.cat(cams)
    cams_avg = cams_avg[:, 0, 1:]
    image_relevance = cams_avg.mean(dim=0).unsqueeze(0)
    image_relevance = image_relevance.reshape(1, 1, 7, 7)
    image_relevance = torch.nn.functional.interpolate(
        image_relevance, size=image_size, mode="bicubic"
    )
    image_relevance = (
        image_relevance.reshape(image_size, image_size)
        .data.cpu()
        .numpy()
        .astype(np.float32)
    )
    image_relevance = (image_relevance - image_relevance.min()) / (
        image_relevance.max() - image_relevance.min()
    )
    return image_relevance


def set_init_strokes_with_attention_map(
    attention_map: torch.Tensor,
    input_image: torch.Tensor,
    num_strokes: int,
    image_size: int,
    xdog_intersec: bool = True,
    mask: torch.Tensor = None,
    tau_max_min: tuple = None,
):
    """
    Initialize stroke positions based on attention map.
    
    Args:
        attention_map: CLIP attention map.
        input_image: Input image tensor.
        num_strokes: Number of strokes to initialize.
        image_size: Image size.
        xdog_intersec: Whether to use XDoG intersection.
        mask: Optional mask tensor.
        tau_max_min: Tuple of (tau_max, tau_min) for sampling.
        
    Returns:
        Tuple of:
        - attn_map_soft_list: Softmax attention maps
        - bg_attn_map_soft_list: Background attention maps
        - inds: Sampled indices
        - inds_normalised: Normalized indices
        - bg_inds: Background indices
        - bg_inds_normalised: Normalized background indices
    """
    # Set default value.
    if tau_max_min is None:
        tau_max_min = (0.3, 0.3)
    taus_list = np.linspace(tau_max_min[0], tau_max_min[1], num_strokes)

    attn_map = (attention_map - attention_map.min()) / (
        attention_map.max() - attention_map.min()
    )
    background_attn_map = 1 - attn_map
    if xdog_intersec:
        xdog = XDoG()
        im_xdog = xdog(input_image.permute(1, 2, 0).cpu().numpy(), k=10)
        im_xdog = resize(im_xdog, (image_size, image_size))
        intersec_map = (1 - im_xdog) * attn_map
        attn_map = intersec_map

    if mask is not None:
        attn_map = attn_map * mask[0].cpu().numpy()
        background_attn_map = background_attn_map * (1 - mask[0].cpu().numpy())

    sampled_inds_list = []
    sampled_bg_inds_list = []
    attn_map_soft_list = []
    bg_attn_map_soft_list = []
    for shape_idx, tau in enumerate(taus_list):
        attn_map_soft = np.copy(attn_map)
        background_attn_map_soft = np.copy(background_attn_map)
        # Use higher float representation to avoid overflow in exp
        attn_map_soft[attn_map > 0] = softmax(
            attn_map[attn_map > 0].astype(np.float128), tau=tau
        ).astype(attn_map.dtype)

        background_attn_map_soft[background_attn_map > 0] = softmax(
            background_attn_map[background_attn_map > 0].astype(np.float128), tau=tau
        ).astype(background_attn_map.dtype)

        idx = np.random.choice(
            range(attn_map.flatten().shape[0]),
            size=1,
            replace=False,
            p=attn_map_soft.flatten(),
        )
        bg_idx = np.random.choice(
            range(background_attn_map_soft.flatten().shape[0]),
            size=1,
            replace=False,
            p=background_attn_map_soft.flatten(),
        )
        sampled_inds_list.append(idx)
        sampled_bg_inds_list.append(bg_idx)
        attn_map_soft_list.append(attn_map_soft)
        bg_attn_map_soft_list.append(background_attn_map_soft)

    inds, inds_normalised = post_process_sampled_indices(
        sampled_inds_list=sampled_inds_list, attn_map=attn_map, image_size=image_size
    )
    bg_inds, bg_inds_normalised = post_process_sampled_indices(
        sampled_inds_list=sampled_bg_inds_list,
        attn_map=background_attn_map,
        image_size=image_size,
    )

    return (
        attn_map_soft_list,
        bg_attn_map_soft_list,
        inds,
        inds_normalised,
        bg_inds,
        bg_inds_normalised,
    )


def post_process_sampled_indices(*, sampled_inds_list, attn_map, image_size):
    """Post-process sampled indices to get normalized coordinates."""
    inds = np.array(sampled_inds_list).flatten()
    inds = np.array(np.unravel_index(inds, attn_map.shape)).T

    inds_normalised = np.zeros(inds.shape)
    inds_normalised[:, 0] = inds[:, 1] / image_size
    inds_normalised[:, 1] = inds[:, 0] / image_size
    inds_normalised = inds_normalised.tolist()

    return inds, inds_normalised


class XDoG:
    """Extended Difference of Gaussians filter for edge detection."""

    def __init__(self):
        super(XDoG, self).__init__()
        self.gamma = 0.98
        self.phi = 200
        self.eps = -0.1
        self.sigma = 0.8
        self.binarize = True

    def __call__(self, im, k=10):
        if im.shape[2] == 3:
            im = rgb2gray(im)
        imf1 = gaussian_filter(im, self.sigma)
        imf2 = gaussian_filter(im, self.sigma * k)
        imdiff = imf1 - self.gamma * imf2
        imdiff = (imdiff < self.eps) * 1.0 + (imdiff >= self.eps) * (
            1.0 + np.tanh(self.phi * imdiff)
        )
        imdiff -= imdiff.min()
        imdiff /= imdiff.max()
        if self.binarize:
            th = threshold_otsu(imdiff)
            imdiff = imdiff >= th
        imdiff = imdiff.astype("float32")
        return imdiff


def softmax(x, tau=0.2):
    """Softmax with temperature scaling."""
    e_x = np.exp(x / tau)
    return e_x / e_x.sum()

