"""
Utility functions for sketch processing.

This module provides helper functions for reading, rendering, and
manipulating SVG sketches.
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from xml.etree import ElementTree as ET

import numpy as np
import pydiffvg
import torch
from PIL import Image, ImageDraw, ImageFont


def read_svg(
    svg_path: Union[str, Path],
    device: torch.device,
    multiply: bool = True,
    canvas_size: int = 512,
    **kwargs,
) -> torch.Tensor:
    """
    Read and render an SVG file to a tensor.
    
    Args:
        svg_path: Path to the SVG file.
        device: Torch device.
        multiply: Whether to apply alpha multiplication.
        canvas_size: Target canvas size.
        
    Returns:
        Rendered image tensor of shape (H, W, 3).
    """
    svg_path = str(svg_path)
    
    if not os.path.exists(svg_path):
        raise FileNotFoundError(f"SVG file not found: {svg_path}")
    
    # Load SVG
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg_path)
    
    # Ensure shapes are on correct device
    for shape in shapes:
        if hasattr(shape, 'points'):
            shape.points = shape.points.to(device)
        if hasattr(shape, 'stroke_width'):
            if torch.is_tensor(shape.stroke_width):
                shape.stroke_width = shape.stroke_width.to(device)
        if hasattr(shape, 'num_control_points'):
            shape.num_control_points = shape.num_control_points.to(device)
    
    for group in shape_groups:
        if group.fill_color is not None:
            group.fill_color = group.fill_color.to(device)
        if group.stroke_color is not None:
            group.stroke_color = group.stroke_color.to(device)
    
    # Render
    _render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups
    )
    img = _render(
        canvas_width, canvas_height,
        2, 2, 0, None,
        *scene_args
    )
    
    # Apply alpha multiplication for white background
    if multiply:
        opacity = img[:, :, 3:4]
        img = opacity * img[:, :, :3] + torch.ones(
            img.shape[0], img.shape[1], 3, device=device
        ) * (1 - opacity)
    else:
        img = img[:, :, :3]
    
    # Resize if needed
    if img.shape[0] != canvas_size or img.shape[1] != canvas_size:
        img = img.permute(2, 0, 1).unsqueeze(0)  # HWC -> NCHW
        img = torch.nn.functional.interpolate(
            img, size=(canvas_size, canvas_size), mode='bilinear', align_corners=False
        )
        img = img.squeeze(0).permute(1, 2, 0)  # NCHW -> HWC
    
    return img

def resize_svg(
    painter,
    target_width: int,
    target_height: int,
) -> None:
    """
    Resize SVG by scaling control points.
    
    Args:
        painter: Painter instance with shapes to resize.
        target_width: Target canvas width.
        target_height: Target canvas height.
    """
    current_width = painter.canvas_size
    current_height = painter.canvas_size
    
    scale_x = target_width / current_width
    scale_y = target_height / current_height
    
    for shape in painter.shapes:
        if hasattr(shape, 'points'):
            shape.points[:, 0] *= scale_x
            shape.points[:, 1] *= scale_y
    
    painter.canvas_size = target_width

def extract_stroke_color(path_elem) -> Optional[str]:
    """
    Extract stroke color from a path element.
    Checks both stroke attribute and style attribute.
    
    Returns:
        Normalized color string (rgb(...) format) or None.
    """
    # Check direct stroke attribute first
    stroke = path_elem.get('stroke', '')
    
    # Check style attribute if stroke not found
    if not stroke:
        style = path_elem.get('style', '')
        if style:
            # Try to extract stroke from style
            style_match = re.search(r'stroke:\s*rgb\([^)]+\)', style)
            if style_match:
                stroke = style_match.group(0).replace('stroke:', '').strip()
    
    # Normalize color string (remove spaces, handle rgb() format)
    if stroke and stroke.startswith('rgb('):
        color_normalized = re.sub(r'\s+', '', stroke)
        return color_normalized
    
    return None

def extract_phase_svg(svg_path: str, phase_idx: int, output_path: str) -> bool:
    """
    Extract paths from a specific phase (by color index) from sketch_group.svg.
    
    Args:
        svg_path: Path to the source SVG file.
        phase_idx: 0-based index (0=phase1, 1=phase2, etc.)
        output_path: Path to write the extracted SVG.
        
    Returns:
        True if successful, False otherwise.
    """
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # Handle namespace
        ns_uri = None
        if root.tag.startswith('{'):
            ns_uri = root.tag[1:root.tag.index('}')]
            ET.register_namespace('', ns_uri)
        
        # Get all colors in order
        colors_seen = set()
        colors_order = []
        
        for path in root.iter():
            tag = path.tag
            # Remove namespace prefix if present
            if ns_uri and tag.startswith('{'):
                tag = tag[len('{' + ns_uri + '}'):]
            
            if tag == 'path':
                color = extract_stroke_color(path)
                if color and color not in colors_seen:
                    colors_seen.add(color)
                    colors_order.append(color)
        
        if phase_idx >= len(colors_order):
            print(f"Warning: Phase index {phase_idx} exceeds available phases ({len(colors_order)})")
            return False
        
        target_color = colors_order[phase_idx]
        
        # Create new SVG structure
        if ns_uri:
            new_root = ET.Element(f"{{{ns_uri}}}svg")
        else:
            new_root = ET.Element("svg")
        
        # Copy attributes from original
        for attr, value in root.attrib.items():
            new_root.set(attr, value)
        
        # Create defs and g elements
        defs = ET.SubElement(new_root, "defs") if ns_uri is None else ET.SubElement(new_root, f"{{{ns_uri}}}defs")
        g = ET.SubElement(new_root, "g") if ns_uri is None else ET.SubElement(new_root, f"{{{ns_uri}}}g")
        
        # Extract paths with target color
        for path in root.iter():
            # Check if this is a path element
            tag = path.tag
            if ns_uri and tag.startswith('{'):
                tag = tag[len('{' + ns_uri + '}'):]
            
            if tag == 'path':
                color = extract_stroke_color(path)
                if color == target_color:
                    # Copy this path element
                    new_path = ET.SubElement(g, tag if ns_uri is None else f"{{{ns_uri}}}{tag}")
                    for attr, value in path.attrib.items():
                        new_path.set(attr, value)
        
        # Write the new SVG
        tree = ET.ElementTree(new_root)
        ET.indent(tree, space="  ")
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        return True
        
    except Exception as e:
        print(f"Warning: Could not extract phase SVG from {svg_path}: {e}")
        return False

def create_phase_grid_with_diffs(
    output_dir: Union[str, Path],
    phases: List[Dict],
    img_size: int = 200,
    padding: int = 10,
) -> Optional[str]:
    """
    Create phase grid image similar to phasegrid single mode.
    Shows: sketch_p1, sketch_p2, ..., sketch_p{n}, p1, (p2-p1), (p3-p2), ...
    Uses sketch_group.svg to extract phase differences by parsing SVG colors.
    
    Args:
        output_dir: Output directory containing sketch_pN.png files and sketch_group.svg.
        phases: List of phase dictionaries.
        img_size: Size of each image in the grid.
        padding: Padding between images.
        
    Returns:
        Path to the saved grid image, or None if failed.
    """
    output_dir = Path(output_dir).resolve() if isinstance(output_dir, str) else Path(output_dir).resolve()
    num_phases = len(phases)
    
    # Collect image paths: p1, p2, ..., p{n}, then p1, (p2-p1), (p3-p2), ...
    image_paths = []
    
    # Add phase images (p1, p2, ..., p{n})
    for phase_idx in range(num_phases):
        phase_name = f"sketch_p{phase_idx + 1}"
        png_path = output_dir / f"{phase_name}.png"
        if png_path.exists():
            image_paths.append(str(png_path))
        else:
            image_paths.append(None)
    
    # Generate phase difference SVGs from sketch_group.svg
    # Phase diff part: p1, (p2-p1), (p3-p2), ...
    group_svg_path = output_dir / "sketch_group.svg"
    diff_svg_paths = []
    
    if group_svg_path.exists():
        with tempfile.TemporaryDirectory(prefix='phasegrid_diffs_') as temp_dir:
            # First, add p1 (phase 1) - extract from sketch_group.svg
            p1_svg_path = os.path.join(temp_dir, 'phase_0.svg')
            if extract_phase_svg(str(group_svg_path), 0, p1_svg_path):
                # Convert to PNG
                p1_png_path = os.path.join(temp_dir, 'phase_0.png')
                try:
                    subprocess.run([
                        'rsvg-convert',
                        '--background-color', 'white',
                        '--width', str(img_size),
                        '--height', str(img_size),
                        '-o', p1_png_path,
                        p1_svg_path
                    ], check=True, capture_output=True)
                    diff_svg_paths.append(p1_png_path)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    diff_svg_paths.append(None)
            else:
                diff_svg_paths.append(None)
            
            # Then add phase differences: (p2-p1), (p3-p2), etc.
            for diff_idx in range(num_phases - 1):
                # Extract phase difference (p2-p1), (p3-p2), etc.
                # diff_idx=0 means (p2-p1), which is phase 2 (index 1)
                phase_idx = diff_idx + 1
                diff_svg_path = os.path.join(temp_dir, f'phase_{phase_idx}.svg')
                if extract_phase_svg(str(group_svg_path), phase_idx, diff_svg_path):
                    # Convert to PNG
                    diff_png_path = os.path.join(temp_dir, f'phase_{phase_idx}.png')
                    try:
                        subprocess.run([
                            'rsvg-convert',
                            '--background-color', 'white',
                            '--width', str(img_size),
                            '--height', str(img_size),
                            '-o', diff_png_path,
                            diff_svg_path
                        ], check=True, capture_output=True)
                        diff_svg_paths.append(diff_png_path)
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        diff_svg_paths.append(None)
                else:
                    diff_svg_paths.append(None)
            
            image_paths.extend(diff_svg_paths)
            
            # Create grid image (inside with block so temp files are still available)
            num_cols = len(image_paths)
            if num_cols == 0:
                print("Warning: No images to create grid")
                return None
            
            # Calculate grid dimensions
            col_label_height = 30
            grid_width = num_cols * img_size + (num_cols + 1) * padding
            grid_height = img_size + col_label_height + padding * 2
            
            # Create grid image
            grid_img = Image.new('RGB', (grid_width, grid_height), color='white')
            draw = ImageDraw.Draw(grid_img)
            
            # Try to load fonts
            try:
                col_label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except:
                try:
                    col_label_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 16)
                except:
                    col_label_font = ImageFont.load_default()
            
            # Place images and labels
            images_y = padding
            labels_y = images_y + img_size + 5
            
            for col_idx, img_path in enumerate(image_paths):
                x = col_idx * img_size + (col_idx + 1) * padding
                
                # Load and paste image
                if img_path and os.path.exists(img_path):
                    try:
                        img = Image.open(img_path)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img = img.resize((img_size, img_size))
                        grid_img.paste(img, (x, images_y))
                    except Exception as e:
                        print(f"Warning: Could not load image {img_path}: {e}")
                
                # Add column label
                if col_idx < num_phases:
                    label_text = f"sketch_p{col_idx + 1}"
                else:
                    diff_idx = col_idx - num_phases
                    if diff_idx == 0:
                        label_text = "p1"
                    else:
                        label_text = f"(p{diff_idx + 1}-p{diff_idx})"
                
                bbox = draw.textbbox((0, 0), label_text, font=col_label_font)
                text_width = bbox[2] - bbox[0]
                text_x = x + (img_size - text_width) // 2
                draw.text((text_x, labels_y), label_text, fill='black', font=col_label_font)
            
            # Save grid image
            output_path = output_dir / "phase_grid.png"
            grid_img.save(output_path, quality=95)
            print(f"Phase grid with differences saved to {output_path}")
            return str(output_path)
    else:
        # If sketch_group.svg doesn't exist, just use p1.png for phase diff part
        p1_png = output_dir / "sketch_p1.png"
        if p1_png.exists():
            image_paths.append(str(p1_png))
        else:
            image_paths.append(None)
        
        # Create grid image
        num_cols = len(image_paths)
        if num_cols == 0:
            print("Warning: No images to create grid")
            return None
        
        # Calculate grid dimensions
        col_label_height = 30
        grid_width = num_cols * img_size + (num_cols + 1) * padding
        grid_height = img_size + col_label_height + padding * 2
        
        # Create grid image
        grid_img = Image.new('RGB', (grid_width, grid_height), color='white')
        draw = ImageDraw.Draw(grid_img)
        
        # Try to load fonts
        try:
            col_label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            try:
                col_label_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 16)
            except:
                col_label_font = ImageFont.load_default()
        
        # Place images and labels
        images_y = padding
        labels_y = images_y + img_size + 5
        
        for col_idx, img_path in enumerate(image_paths):
            x = col_idx * img_size + (col_idx + 1) * padding
            
            # Load and paste image
            if img_path and os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img = img.resize((img_size, img_size))
                    grid_img.paste(img, (x, images_y))
                except Exception as e:
                    print(f"Warning: Could not load image {img_path}: {e}")
            
            # Add column label
            if col_idx < num_phases:
                label_text = f"sketch_p{col_idx + 1}"
            else:
                diff_idx = col_idx - num_phases
                if diff_idx == 0:
                    label_text = "p1"
                else:
                    label_text = f"(p{diff_idx + 1}-p{diff_idx})"
            
            bbox = draw.textbbox((0, 0), label_text, font=col_label_font)
            text_width = bbox[2] - bbox[0]
            text_x = x + (img_size - text_width) // 2
            draw.text((text_x, labels_y), label_text, fill='black', font=col_label_font)
        
        # Save grid image
        output_path = output_dir / "phase_grid.png"
        grid_img.save(output_path, quality=95)
        print(f"Phase grid with differences saved to {output_path}")
        return str(output_path)

def svg_to_pil(svg_files: List[str], device: torch.device, save_dir: Optional[str] = None, ext: str = 'png'):
    """
    Convert SVG to PIL Image.
    
    Args:
        svg_files: The list of paths to the SVG files.
        save_dir: The directory to save the PIL images. If None, the images will not be saved.
        device: The device to use for the tensor.
        ext: The extension of the image file.
        
    Returns:
        img_pils: The list of PIL images.
        img_paths: The list of image paths if save_dir is not None.
    """
    img_pils = []
    img_paths = []
    for svg_file in svg_files:
        sketch = read_svg(svg_file, device, multiply=True)
        sketch_np = sketch.cpu().numpy()
        sketch_pil = Image.fromarray((sketch_np * 255).astype('uint8'), 'RGB')
        img_pils.append(sketch_pil)
        if save_dir is not None:
            img_path = os.path.join(save_dir, os.path.basename(svg_file).replace('.svg', f'.{ext}'))
            sketch_pil.save(img_path)
            img_paths.append(img_path)
    
    if save_dir is not None:
        return img_pils, img_paths
    else:
        return img_pils

def tensor_to_pil(tensor: torch.Tensor, save_path: Optional[str] = None) -> Image.Image:
    """
    Convert a tensor to PIL Image.
    
    Args:
        tensor: Image tensor (can be CHW or HWC, values in [0, 1] or [0, 255]).
        save_path: The path to save the image.
    Returns:
        PIL Image.
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    if tensor.shape[0] in [1, 3, 4]:
        tensor = tensor.permute(1, 2, 0)
    
    tensor = tensor.detach().cpu().numpy()
    
    if tensor.max() <= 1.0:
        tensor = (tensor * 255).astype(np.uint8)
    else:
        tensor = tensor.astype(np.uint8)
    
    if tensor.shape[2] == 1:
        tensor = tensor.squeeze(2)
        img_pil = Image.fromarray(tensor, mode='L')
    elif tensor.shape[2] == 4:
        img_pil = Image.fromarray(tensor, mode='RGBA')
    else:
        img_pil = Image.fromarray(tensor, mode='RGB')
    
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        img_pil.save(save_path)
    return img_pil

def pil_to_tensor(
    image: Image.Image,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Convert PIL Image to tensor.
    
    Args:
        image: PIL Image.
        device: Target device.
        
    Returns:
        Tensor of shape (H, W, C) with values in [0, 1].
    """
    tensor = torch.from_numpy(np.array(image)).float() / 255.0
    
    if device is not None:
        tensor = tensor.to(device)
    
    return tensor
