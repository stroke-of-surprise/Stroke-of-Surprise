#!/usr/bin/env python3
"""
Script to create grid images from final and first SVG files across multiple run folders.
Creates separate grids for: final non-colored SVGs, final colored SVGs, 
first non-colored SVGs, and first colored SVGs.
"""

import os
import sys
import re
import argparse
from pathlib import Path
import subprocess
from PIL import Image, ImageDraw, ImageFont
import tempfile
import math
import tqdm
import termcolor

def natural_sort_key(s):
    """Sort strings with numbers naturally (e.g., run2 comes before run10)"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]

def convert_to_png_uniform(input_path, png_path, width=512):
    """
    Convert input image to a PNG with a fixed size.
    - If input is SVG: use rsvg-convert (same as before).
    - If input is raster (e.g., PNG/JPEG): use PIL to resize and save.
    """
    ext = str(input_path).lower()
    if ext.endswith(".svg"):
        try:
            subprocess.run([
                'rsvg-convert',
                '--background-color', 'white',
                '--width', str(width),
                '--height', str(width),
                '-o', png_path,
                str(input_path)
            ], check=True, capture_output=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"Warning: Could not convert SVG {input_path}. Please install rsvg-convert (librsvg) or adjust the script.")
            return False
    else:
        try:
            img = Image.open(input_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((width, width))
            img.save(png_path)
            return True
        except Exception as e:
            print(f"Warning: Could not process raster image {input_path}: {e}")
            return False

def create_grid_image(image_files, run_names, output_path, img_size=512, cols=None, padding=10, label_height=40, title=None, title_height=60):
    """
    Create a grid image from SVG files
    
    Args:
        image_files: List of image file paths (SVG or raster images like PNG)
        run_names: List of run folder names (for labels)
        output_path: Output image path
        img_size: Size of each image in the grid
        cols: Number of columns (auto-calculated if None)
        padding: Padding between images
        label_height: Height of label area below each image
        title: Title text to display at the top of the grid
        title_height: Height of title area
    """
    if not image_files:
        print(f"No image files to process for {output_path}")
        return False
    
    n_images = len(image_files)
    
    # Auto-calculate grid dimensions
    if cols is None:
        cols = math.ceil(math.sqrt(n_images))
    rows = math.ceil(n_images / cols)
    
    print(f"Creating {rows}x{cols} grid with {n_images} images: {output_path}")
    
    # Create temporary directory for PNG conversions
    with tempfile.TemporaryDirectory() as temp_dir:
        png_files = []
        
        # Convert each input image to a temporary, uniformly sized PNG
        for i, img_file in tqdm.tqdm(enumerate(image_files), total=len(image_files), desc="Converting images to PNGs"):
            png_path = os.path.join(temp_dir, f'image_{i:04d}.png')
            
            if convert_to_png_uniform(img_file, png_path, img_size):
                png_files.append(png_path)
            else:
                # Create a placeholder image if conversion fails
                placeholder = Image.new('RGB', (img_size, img_size), color='white')
                placeholder.save(png_path)
                png_files.append(png_path)
        
        if not png_files:
            print(f"Error: No PNG files were created for {output_path}")
            return False
        
        # Calculate grid dimensions
        cell_width = img_size
        cell_height = img_size + label_height
        grid_width = cols * cell_width + (cols + 1) * padding
        grid_height = rows * cell_height + (rows + 1) * padding
        
        # Add title height if title is provided
        if title:
            grid_height += title_height
            title_offset = title_height
        else:
            title_offset = 0
        
        # Create grid image
        grid_img = Image.new('RGB', (grid_width, grid_height), color='white')
        draw = ImageDraw.Draw(grid_img)
        
        # Try to load fonts
        try:
            label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
        except:
            try:
                label_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 20)
                title_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 32)
            except:
                label_font = ImageFont.load_default()
                title_font = ImageFont.load_default()
        
        # Draw title if provided
        if title:
            bbox = draw.textbbox((0, 0), title, font=title_font)
            title_width = bbox[2] - bbox[0]
            title_x = (grid_width - title_width) // 2
            draw.text((title_x, 15), title, fill='black', font=title_font)
        
        # Place images in grid
        for idx, (png_file, run_name) in enumerate(zip(png_files, run_names)):
            row = idx // cols
            col = idx % cols
            
            # Calculate position
            x = col * cell_width + (col + 1) * padding
            y = row * cell_height + (row + 1) * padding + title_offset
            
            # Load and paste image
            img = Image.open(png_file)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            grid_img.paste(img, (x, y))
            
            # Add label
            label_y = y + img_size + 5
            # Center the text
            bbox = draw.textbbox((0, 0), run_name, font=label_font)
            text_width = bbox[2] - bbox[0]
            text_x = x + (img_size - text_width) // 2
            draw.text((text_x, label_y), run_name, fill='black', font=label_font)
        
        # Save grid image
        grid_img.save(output_path, quality=95)
        print(f"✓ Created grid: {output_path}")
        
        return True

def main():
    parser = argparse.ArgumentParser(
        description='Create grid images from final SVG files across multiple run folders',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n"
        "  python create_grid_from_runs.py output/loop/jellyfish+horse/2A1B_gp2_depth_loop\n"
        "  python create_grid_from_runs.py output/loop/jellyfish+horse/2A1B_gp2_depth_loop --cols 5 --img-size 400"
    )
    parser.add_argument(
        'folder',
        type=str,
        help='Folder containing subdirectories with final_svg files (e.g., ControlSketch/output/loop/.../2A1B_gp2_depth_loop or ControlSketch/output/illusion_sketch/bicycle+rabbit)'
    )
    parser.add_argument(
        '--img-size',
        type=int,
        default=300,
        help='Size of each image in pixels (default: 512)'
    )
    parser.add_argument(
        '--cols',
        type=int,
        default=None,
        help='Number of columns in grid (auto-calculated if not specified)'
    )
    parser.add_argument(
        '--padding',
        type=int,
        default=10,
        help='Padding between images in pixels (default: 10)'
    )
    parser.add_argument(
        '--n_samples',
        '-n',
        type=int,
        default=None,
        help='Number of samples to include in the grid (default: all)'
    )
    
    args = parser.parse_args()
    
    # Get the folder path
    folder = Path(args.folder)
    if not folder.exists():
        print(f"Error: Folder does not exist: {folder}")
        sys.exit(1)
    
    # Find all folders that might contain sketch_pN files
    # First check if there are subfolders in each folder
    candidate_folders = sorted([d for d in folder.iterdir() if d.is_dir() and not d.name.startswith('.')],
                              key=lambda x: natural_sort_key(x.name))
    
    if not candidate_folders:
        print(f"Error: No folders found in: {folder}")
        sys.exit(1)
    
    print(f"Found {len(candidate_folders)} folders: {', '.join([d.name for d in candidate_folders])}")
    
    # For new structure: each run folder contains files like:
    #   sketch_p1.png, sketch_p1.svg
    #   sketch_p1_black.png, sketch_p1_black.svg (optional)
    #   sketch_p2.png, sketch_p2.svg
    #   sketch_p2_black.png, sketch_p2_black.svg (optional)
    #   sketch_p2_group.png, sketch_p2_group.svg (optional)
    #   ...
    #
    # We want: for each existing p-index (1,2,3,...) and variant (normal/black/group),
    # create one comparison grid across runs, similar to the previous behavior.

    print("\n" + "="*60)
    print("Collecting sketch_pN files from folders...")
    print("="*60)

    # List of run folders and labels (folder names)
    run_folders = []
    labels = []

    # Per-run mappings:
    #  - per_run_p_maps[i]: dict[(p_idx, variant)] = Path
    #  - per_run_group_paths[i]: Path or None for sketch_group.(png|svg)
    per_run_p_maps = []
    per_run_group_paths = []

    sketch_p_pattern = re.compile(r"^sketch(_p(\d+))?(?:_(black))?\.(png|svg)$", re.IGNORECASE)
    sketch_group_pattern = re.compile(r"^sketch_group\.(png|svg)$", re.IGNORECASE)

    for folder_item in candidate_folders:
        if not folder_item.is_dir():
            continue

        run_p_map = {}
        run_group_path = None

        for file_path in folder_item.iterdir():
            if not file_path.is_file():
                continue
            name = file_path.name

            # First check group image
            if sketch_group_pattern.match(name):
                if run_group_path is None:
                    run_group_path = file_path
                else:
                    # Prefer PNG over SVG if both exist
                    existing_ext = str(run_group_path).lower()
                    current_ext = str(file_path).lower()
                    if existing_ext.endswith(".svg") and current_ext.endswith(".png"):
                        run_group_path = file_path
                continue

            # Then check sketch[_pN][_black]
            m_p = sketch_p_pattern.match(name)
            if not m_p:
                continue
            
            _, p_idx_str, variant, _ext = m_p.groups()
            # If no _p index, treat as 0 (single object case)
            p_idx = int(p_idx_str) if p_idx_str is not None else 0
            variant_key = variant if variant is not None else "normal"
            key = (p_idx, variant_key)

            # Prefer PNG over SVG if both exist for the same key
            if key in run_p_map:
                existing_ext = str(run_p_map[key]).lower()
                current_ext = str(file_path).lower()
                if existing_ext.endswith(".svg") and current_ext.endswith(".png"):
                    run_p_map[key] = file_path
            else:
                run_p_map[key] = file_path

        # Only keep runs that actually have at least one sketch_pN file or a group image
        if run_p_map or run_group_path is not None:
            run_folders.append(folder_item)
            labels.append(folder_item.name)
            per_run_p_maps.append(run_p_map)
            per_run_group_paths.append(run_group_path)

    if not run_folders:
        print("Error: No valid sketch_pN files found in any subfolder!")
        sys.exit(1)

    print(f"\nCollected {len(labels)} folders with sketch_pN files.")
    if len(labels) <= 10:
        print(f"Labels: {', '.join(labels)}")
    else:
        print(f"Labels: {', '.join(labels[:5])}, ... , {', '.join(labels[-5:])}")

    # Apply n_samples limit if specified (limit number of runs)
    if args.n_samples is not None and args.n_samples > 0:
        original_count = len(labels)
        if original_count > args.n_samples:
            print(f"\nLimiting to first {args.n_samples} samples (out of {original_count} total)")
            run_folders = run_folders[:args.n_samples]
            labels = labels[:args.n_samples]
            per_run_p_maps = per_run_p_maps[:args.n_samples]
            per_run_group_paths = per_run_group_paths[:args.n_samples]

    # Collect all (p_idx, variant) keys across runs (for sketch_pN files)
    all_p_keys = set()
    for run_map in per_run_p_maps:
        all_p_keys.update(run_map.keys())

    if not all_p_keys and not any(per_run_group_paths):
        print("Error: No sketch_pN or sketch_group files discovered after scanning runs!")
        sys.exit(1)

    # Sort keys: first by p_idx, then by variant order (normal, black, group, others alphabetically)
    def variant_order(v):
        if v == "normal":
            return 0
        if v == "black":
            return 1
        if v == "group":
            return 2
        return 10

    sorted_p_keys = sorted(all_p_keys, key=lambda kv: (kv[0], variant_order(kv[1]), kv[1]))

    # Create grids for each sketch_pN (p_idx, variant) and for sketch_group
    output_dir = folder
    title_base = str(folder)

    print("\n" + "="*60)
    print("Creating grids for each sketch_pN / variant...")
    print("="*60)

    all_grids_created = []

    # First: grids for sketch_pN variants
    for (p_idx, variant) in sorted_p_keys:
        # Collect image paths and labels for this key across runs
        image_paths = []
        label_for_image = []
        for run_label, run_map in zip(labels, per_run_p_maps):
            img_path = run_map.get((p_idx, variant))
            if img_path is not None:
                image_paths.append(img_path)
                label_for_image.append(run_label)

        if not image_paths:
            continue

        # Determine output file name
        if variant == "normal":
            suffix = ""
        else:
            suffix = f"_{variant}"
        
        if p_idx == 0:
            out_name = f"grid_sketch{suffix}.png"
            display_name = f"sketch{suffix}"
        else:
            out_name = f"grid_sketch_p{p_idx}{suffix}.png"
            display_name = f"sketch_p{p_idx}{suffix}"
            
        out_path = output_dir / out_name
        title = f"{title_base} | {display_name}"

        print("\n" + "-"*60)
        print(f"Creating grid for {display_name} ({len(image_paths)} images)")
        print("-"*60)

        success = create_grid_image(
            image_paths,
            label_for_image,
            str(out_path),
            args.img_size,
            args.cols,
            args.padding,
            title=title
        )

        if success:
            all_grids_created.append(str(out_path))

    # Then: one grid for sketch_group (if any run has it)
    group_image_paths = []
    group_labels = []
    for run_label, group_path in zip(labels, per_run_group_paths):
        if group_path is not None:
            group_image_paths.append(group_path)
            group_labels.append(run_label)

    if group_image_paths:
        out_name = "grid_sketch_group.png"
        out_path = output_dir / out_name
        title = f"{title_base} | sketch_group"

        print("\n" + "-"*60)
        print(f"Creating grid for sketch_group ({len(group_image_paths)} images)")
        print("-"*60)

        success_group = create_grid_image(
            group_image_paths,
            group_labels,
            str(out_path),
            args.img_size,
            args.cols,
            args.padding,
            title=title
        )

        if success_group:
            all_grids_created.append(str(out_path))

    # Final summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    if all_grids_created:
        for p in all_grids_created:
            print(termcolor.colored(f"✓ Grid: {p}", "green"))
        print(f"\nTotal: {len(all_grids_created)} grid image(s) created")
    else:
        print("Error: No grids were created!")
        sys.exit(1)

    print("\nAll done!")
    print("="*60)

if __name__ == '__main__':
    main()

