#!/usr/bin/env python3
"""
Script to create grid images where each row represents one run.
Each row shows: sketch_p1, sketch_p2, ..., sketch_p{n}, (p2-p1), (p3-p2), ..., intersection (if exists)
Phase differences are extracted from sketch_group.svg by parsing paths by color.
If intersect_logs/intersection.png exists, it will be displayed as the last column in each row.
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
from xml.etree import ElementTree as ET
import termcolor
import json

def natural_sort_key(s):
    """Sort strings with numbers naturally (e.g., run2 comes before run10)"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]

def extract_stroke_color(path_elem):
    """
    Extract stroke color from a path element.
    Checks both stroke attribute and style attribute.
    Returns normalized color string (rgb(...) format) or None.
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

def parse_svg_colors(svg_path):
    """
    Parse SVG file and extract unique stroke colors.
    Returns a list of unique color strings in order of appearance.
    """
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        colors_seen = set()
        colors_order = []
        
        # Find all path elements
        for path in root.iter():
            tag = path.tag
            # Remove namespace prefix if present
            if '}' in tag:
                tag = tag.split('}')[-1]
            
            if tag == 'path':
                color = extract_stroke_color(path)
                if color and color not in colors_seen:
                    colors_seen.add(color)
                    colors_order.append(color)
        
        return colors_order
    except Exception as e:
        print(f"Warning: Could not parse SVG colors from {svg_path}: {e}")
        return []

def extract_phase_svg(svg_path, phase_idx, output_path):
    """
    Extract paths from a specific phase (by color index) from sketch_group.svg.
    phase_idx: 0-based index (0=phase1, 1=phase2, etc.)
    """
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # Handle namespace
        ns_uri = None
        if root.tag.startswith('{'):
            ns_uri = root.tag[1:root.tag.index('}')]
            ns = {'svg': ns_uri}
            ET.register_namespace('', ns_uri)
        else:
            ns = {'svg': 'http://www.w3.org/2000/svg'}
        
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

def convert_to_png_uniform(input_path, png_path, width=512):
    """
    Convert input image to a PNG with a fixed size.
    - If input is SVG: use rsvg-convert.
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
            print(f"Warning: Could not convert SVG {input_path}. Please install rsvg-convert (librsvg).")
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

def create_phase_grid_image(run_data_list, output_path, img_size=512, padding=10, label_height=40, title=None, title_height=60, col_label_height=30, show_row_titles=True):
    """
    Create a grid image where each row represents one run.
    Each row shows: sketch_p1, sketch_p2, ..., sketch_p{n}, p1, (p2-p1), (p3-p2), ..., intersection (if exists)
    
    Args:
        run_data_list: List of tuples (run_label, image_paths_list, has_group_svg, group_svg_path, run_folder)
                      where image_paths_list contains paths for p1, p2, p3, ..., phase differences, and optionally intersection.png
                      run_folder is the Path to the run folder (for reading config.json)
        output_path: Output image path
        img_size: Size of each image in the grid
        padding: Padding between images
        label_height: Height of label area below each image row
        title: Title text to display at the top
        title_height: Height of title area
        col_label_height: Height for column labels
        show_row_titles: Whether to show row titles (run names) above each row
    """
    if not run_data_list:
        print(f"No run data to process for {output_path}")
        return False
    
    n_runs = len(run_data_list)
    
    # Find maximum number of columns across all runs and determine phase count
    # Structure: [p1, p2, ..., p{k}, p1, (p2-p1), (p3-p2), ..., (p{k}-p{k-1})]
    # So if we have k phases, we have: k phase images + k phase diff images = 2k images
    max_cols = 0
    num_phases = 0
    for run_data in run_data_list:
        # Handle both old format (4 elements) and new format (5 elements)
        if len(run_data) == 4:
            run_label, image_paths, has_group_svg, _ = run_data
        else:
            run_label, image_paths, has_group_svg, _, _ = run_data
        max_cols = max(max_cols, len(image_paths))
        if has_group_svg and len(image_paths) > 0:
            # Structure: k phases + k phase diffs = 2k images
            # So num_phases = total_images // 2
            num_phases = max(num_phases, len(image_paths) // 2)
    
    # If we couldn't determine from group_svg, infer from column count
    if num_phases == 0 and max_cols > 0:
        num_phases = max_cols // 2
    
    if max_cols == 0:
        print(f"Error: No images to display for {output_path}")
        return False
    
    print(f"Creating grid with {n_runs} rows, up to {max_cols} columns ({num_phases} phases): {output_path}")
    
    # Create temporary directory for PNG conversions
    with tempfile.TemporaryDirectory() as temp_dir:
        all_row_images = []
        
        for run_idx, run_data in enumerate(run_data_list):
            # Handle both old format (4 elements) and new format (5 elements)
            if len(run_data) == 4:
                run_label, image_paths, has_group_svg, group_svg_path = run_data
                run_folder = None
            else:
                run_label, image_paths, has_group_svg, group_svg_path, run_folder = run_data
            png_files = []
            
            # Convert each image to PNG
            for img_idx, img_file in enumerate(image_paths):
                png_path = os.path.join(temp_dir, f'run_{run_idx}_img_{img_idx:04d}.png')
                
                # Convert Path object to string if needed
                img_file_str = str(img_file) if img_file else None
                
                if img_file_str and os.path.exists(img_file_str):
                    if convert_to_png_uniform(img_file_str, png_path, img_size):
                        png_files.append(png_path)
                    else:
                        # Create placeholder
                        placeholder = Image.new('RGB', (img_size, img_size), color='lightgray')
                        placeholder.save(png_path)
                        png_files.append(png_path)
                else:
                    # Create placeholder for missing images
                    placeholder = Image.new('RGB', (img_size, img_size), color='lightgray')
                    placeholder.save(png_path)
                    png_files.append(png_path)
            
            all_row_images.append((run_label, png_files))
        
        # Calculate grid dimensions
        # Each row includes: (optional row_title_height) + img_size + col_label_height
        row_title_height = 30 if show_row_titles else 0  # Height for row title (run name) above each row
        cell_width = img_size
        # Row height: (optional title) + image + column labels
        row_height = row_title_height + img_size + col_label_height
        grid_width = max_cols * cell_width + (max_cols + 1) * padding
        # Total height: title + rows with spacing
        grid_height = n_runs * row_height + (n_runs + 1) * padding
        
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
            col_label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            try:
                label_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 20)
                title_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 32)
                col_label_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 16)
            except:
                label_font = ImageFont.load_default()
                title_font = ImageFont.load_default()
                col_label_font = ImageFont.load_default()
        
        # Draw title if provided
        if title:
            bbox = draw.textbbox((0, 0), title, font=title_font)
            title_width = bbox[2] - bbox[0]
            title_x = (grid_width - title_width) // 2
            draw.text((title_x, 15), title, fill='black', font=title_font)
        
        # Place images in grid with row titles, borders, and column labels
        for row_idx, (run_label, png_files) in enumerate(all_row_images):
            # Get run_folder from run_data_list for reading config.json
            run_data = run_data_list[row_idx]
            run_folder = run_data[4] if len(run_data) > 4 else None
            
            row_start_y = title_offset + row_idx * row_height + (row_idx + 1) * padding
            row_images_y = row_start_y + row_title_height
            row_col_labels_y = row_images_y + img_size + 5
            
            # Calculate row width based on actual number of columns in this row
            row_width = len(png_files) * cell_width + (len(png_files) + 1) * padding
            
            # Draw row title (run name) at the top of the row - centered, no border (only if show_row_titles)
            if show_row_titles:
                row_title_y = row_start_y + 5
                bbox = draw.textbbox((0, 0), run_label, font=label_font)
                text_width = bbox[2] - bbox[0]
                # Center the title over the entire row width
                text_x = padding + (row_width - text_width) // 2
                draw.text((text_x, row_title_y), run_label, fill='black', font=label_font)
            
            # Place images in this row
            for col_idx, png_file in enumerate(png_files):
                x = col_idx * cell_width + (col_idx + 1) * padding
                
                # Load and paste image
                img = Image.open(png_file)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                grid_img.paste(img, (x, row_images_y))
            
            # Draw column labels below images for this row
            # First part: sketch_p1, sketch_p2, ..., sketch_p{n}
            for col_idx in range(min(num_phases, len(png_files))):
                x = col_idx * cell_width + (col_idx + 1) * padding
                label_text = f"sketch_p{col_idx + 1}"
                bbox = draw.textbbox((0, 0), label_text, font=col_label_font)
                text_width = bbox[2] - bbox[0]
                text_x = x + (img_size - text_width) // 2
                draw.text((text_x, row_col_labels_y), label_text, fill='black', font=col_label_font)
            
            # Second part: p1, (p2-p1), (p3-p2), ...
            phase_diff_start_col = num_phases
            phase_diff_end_col = phase_diff_start_col + num_phases  # Expected end of phase differences
            for col_idx in range(phase_diff_start_col, len(png_files)):
                x = col_idx * cell_width + (col_idx + 1) * padding
                # Check if this is the intersection column (last column beyond phase differences)
                if col_idx >= phase_diff_end_col:
                    # Read intersection_count from config.json if available
                    intersection_count = None
                    if run_folder:
                        config_path = Path(run_folder) / "config.json"
                        if config_path.exists():
                            try:
                                with open(config_path, 'r') as f:
                                    config = json.load(f)
                                    intersection_count = config.get('intersection_count')
                            except Exception:
                                pass
                    
                    if intersection_count is not None:
                        label_text = f"intersection ({intersection_count})"
                    else:
                        label_text = "intersection"
                else:
                    diff_idx = col_idx - phase_diff_start_col
                    if diff_idx == 0:
                        label_text = "p1"
                    else:
                        # diff_idx=1 -> (p2-p1), diff_idx=2 -> (p3-p2), etc.
                        label_text = f"(p{diff_idx + 1}-p{diff_idx})"
                bbox = draw.textbbox((0, 0), label_text, font=col_label_font)
                text_width = bbox[2] - bbox[0]
                text_x = x + (img_size - text_width) // 2
                draw.text((text_x, row_col_labels_y), label_text, fill='black', font=col_label_font)
            
            # Draw separator line below this row (except for the last row)
            if row_idx < len(all_row_images) - 1:
                row_bottom_y = row_start_y + row_height
                # Draw horizontal line across the entire grid width
                draw.line(
                    [padding, row_bottom_y, row_width - padding, row_bottom_y],
                    fill='black', width=1
                )
        
        # Save grid image
        grid_img.save(output_path, quality=95)
        print(termcolor.colored(f"✓ Created grid: {output_path}", "green"))
        
        return True

def main():
    parser = argparse.ArgumentParser(
        description='Create grid images where each row represents one run, showing phases and phase differences',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n"
        "  python phasegrid.py ControlSketch/test/apple+chicken+angel\n"
        "  python phasegrid.py ControlSketch/test/apple+chicken+angel --img-size 300 --rows 10"
    )
    parser.add_argument(
        'folder',
        type=str,
        help='Folder containing subdirectories with sketch_pN files and sketch_group.svg, or a single run folder with sketch_pN files directly'
    )
    parser.add_argument(
        '--img-size',
        type=int,
        default=200,
        help='Size of each image in pixels (default: 200)'
    )
    parser.add_argument(
        '--padding',
        type=int,
        default=10,
        help='Padding between images in pixels (default: 10)'
    )
    parser.add_argument(
        '--rows',
        type=int,
        default=None,
        help='Maximum number of rows (runs) per output image (default: all in one image)'
    )
    
    args = parser.parse_args()
    
    # Get the folder path
    folder = Path(args.folder)
    if not folder.exists():
        print(f"Error: Folder does not exist: {folder}")
        sys.exit(1)
    
    # Pattern to match sketch_pN files
    sketch_p_pattern = re.compile(r"^sketch_p(\d+)\.(png|svg)$", re.IGNORECASE)
    sketch_group_pattern = re.compile(r"^sketch_group\.svg$", re.IGNORECASE)
    
    # Check if the folder itself contains sketch_pN files (single run folder)
    has_sketch_files = False
    for file_path in folder.iterdir():
        if file_path.is_file():
            name = file_path.name
            if sketch_p_pattern.match(name) or sketch_group_pattern.match(name):
                has_sketch_files = True
                break
    
    if has_sketch_files:
        # This is a single run folder - use it directly
        candidate_folders = [folder]
        print(f"Treating as single run folder: {folder.name}")
    else:
        # Find all run folders (subdirectories)
        candidate_folders = sorted([d for d in folder.iterdir() if d.is_dir() and not d.name.startswith('.')],
                                  key=lambda x: natural_sort_key(x.name))
        
        if not candidate_folders:
            print(f"Error: No folders or sketch_pN files found in: {folder}")
            sys.exit(1)
        
        print(f"Found {len(candidate_folders)} folders")
    
    # Create a persistent temporary directory for phase difference SVGs
    # This will be cleaned up at the end
    temp_diff_base = tempfile.mkdtemp(prefix='phasegrid_diff_')
    
    try:
        # Collect data for each run
        all_run_data = []
        
        for run_idx, run_folder in enumerate(tqdm.tqdm(candidate_folders, desc="Scanning runs")):
            if not run_folder.is_dir():
                continue
            
            # Find all sketch_pN files
            phase_files = {}  # p_idx -> (png_path, svg_path)
            group_svg_path = None
            
            for file_path in run_folder.iterdir():
                if not file_path.is_file():
                    continue
                
                name = file_path.name
                
                # Check for sketch_group.svg
                if sketch_group_pattern.match(name):
                    group_svg_path = file_path
                    continue
                
                # Check for sketch_pN files
                m = sketch_p_pattern.match(name)
                if m:
                    p_idx = int(m.group(1))
                    ext = m.group(2).lower()
                    
                    if p_idx not in phase_files:
                        phase_files[p_idx] = (None, None)
                    
                    if ext == 'png':
                        phase_files[p_idx] = (file_path, phase_files[p_idx][1])
                    elif ext == 'svg':
                        phase_files[p_idx] = (phase_files[p_idx][0], file_path)
            
            # Skip if no phase files found
            if not phase_files:
                continue
            
            # Build image list: p1, p2, p3, ..., p{n}, (p2-p1), (p3-p2), ..., intersection (if exists)
            image_paths = []
            
            # Add phase images (p1, p2, p3, ...)
            sorted_p_indices = sorted(phase_files.keys())
            for p_idx in sorted_p_indices:
                png_path, svg_path = phase_files[p_idx]
                # Prefer PNG, fallback to SVG
                if png_path and png_path.exists():
                    image_paths.append(png_path)
                elif svg_path and svg_path.exists():
                    image_paths.append(svg_path)
                else:
                    image_paths.append(None)
            
            # Generate phase difference SVGs if group_svg exists
            # Phase diff part: p1, (p2-p1), (p3-p2), ...
            if group_svg_path and group_svg_path.exists():
                diff_svgs = []
                # First, add p1 (phase 1)
                p1_svg_path = os.path.join(temp_diff_base, f'run_{run_idx}_phase_0.svg')
                if extract_phase_svg(group_svg_path, 0, p1_svg_path):
                    diff_svgs.append(Path(p1_svg_path))
                else:
                    diff_svgs.append(None)
                
                # Then add phase differences: (p2-p1), (p3-p2), etc.
                for diff_idx in range(len(sorted_p_indices) - 1):
                    # Extract phase difference (p2-p1), (p3-p2), etc.
                    # diff_idx=0 means (p2-p1), which is phase 2 (index 1)
                    phase_idx = diff_idx + 1
                    diff_svg_path = os.path.join(temp_diff_base, f'run_{run_idx}_phase_{phase_idx}.svg')
                    if extract_phase_svg(group_svg_path, phase_idx, diff_svg_path):
                        diff_svgs.append(Path(diff_svg_path))
                    else:
                        diff_svgs.append(None)
                
                # Add phase differences to image_paths
                image_paths.extend(diff_svgs)
            
            # Check for intersection.png in intersect_logs subdirectory
            intersect_logs_dir = run_folder / "intersect_logs"
            intersection_path = intersect_logs_dir / "intersection.png" if intersect_logs_dir.exists() else None
            if intersection_path and intersection_path.exists():
                image_paths.append(intersection_path)
            
            all_run_data.append((run_folder.name, image_paths, group_svg_path is not None, group_svg_path, run_folder))
    
        if not all_run_data:
            print("Error: No valid run data found!")
            sys.exit(1)
        
        print(f"\nCollected {len(all_run_data)} runs")
        
        # Split into batches if --rows is specified
        output_dir = folder
        title_base = str(folder)
        
        # Determine if this is a single run (show_row_titles=False, use run label as title)
        is_single_run = len(all_run_data) == 1
        show_row_titles = not is_single_run
        
        if is_single_run:
            # Single run: use run label as title, don't show row titles
            run_label = all_run_data[0][0]
            out_path = output_dir / "phase_grid.png"
            title = run_label
            
            print(f"\nCreating grid for single run: {run_label}")
            
            create_phase_grid_image(
                all_run_data,
                str(out_path),
                args.img_size,
                args.padding,
                title=title,
                show_row_titles=show_row_titles
            )
        elif args.rows and args.rows > 0:
            # Split into multiple images
            num_batches = math.ceil(len(all_run_data) / args.rows)
            for batch_idx in range(num_batches):
                start_idx = batch_idx * args.rows
                end_idx = min(start_idx + args.rows, len(all_run_data))
                batch_data = all_run_data[start_idx:end_idx]
                
                out_name = f"phase_grid_{batch_idx + 1}.png"
                
                out_path = output_dir / out_name
                title = f"{title_base} | Rows {start_idx + 1}-{end_idx}"
                
                print(f"\nCreating grid batch {batch_idx + 1}/{num_batches} ({len(batch_data)} runs)...")
                
                create_phase_grid_image(
                    batch_data,
                    str(out_path),
                    args.img_size,
                    args.padding,
                    title=title,
                    show_row_titles=show_row_titles
                )
        else:
            # Single image with all runs
            out_path = output_dir / "phase_grid.png"
            title = f"{title_base} | Phase Grid"
            
            print(f"\nCreating grid with {len(all_run_data)} runs...")
            
            create_phase_grid_image(
                all_run_data,
                str(out_path),
                args.img_size,
                args.padding,
                title=title,
                show_row_titles=show_row_titles
            )
        
        print("\nAll done!")
    finally:
        # Clean up temporary directory
        import shutil
        if os.path.exists(temp_diff_base):
            shutil.rmtree(temp_diff_base)

if __name__ == '__main__':
    main()

