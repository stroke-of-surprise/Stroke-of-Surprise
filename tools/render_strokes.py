#!/usr/bin/env python3
"""
Re-render B-spline strokes from saved pickle file.

Usage examples:
    # Render all strokes in black
    python scripts/render_strokes.py output/experiment-1/strokes.pkl -o render_all.png
    
    # Render only phase 1 strokes (indices 0-5)
    python scripts/render_strokes.py output/experiment-1/strokes.pkl -o render_p1.png --indices 0,1,2,3,4,5
    
    # Render with custom colors (red for phase 1, blue for phase 2)
    python scripts/render_strokes.py output/experiment-1/strokes.pkl -o render_colored.png --phase-colors
    
    # Render specific strokes with specific colors
    python scripts/render_strokes.py output/experiment-1/strokes.pkl -o render_custom.png \
        --indices 0,1,2 --colors "1.0,0,0;0,1.0,0;0,0,1.0"
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import matplotlib.cm as cm


def parse_indices(indices_str: str) -> list:
    """Parse comma-separated indices string."""
    if not indices_str:
        return None
    return [int(i.strip()) for i in indices_str.split(',')]


def parse_colors(colors_str: str) -> list:
    """Parse colors string. Format: 'r,g,b;r,g,b;...' """
    if not colors_str:
        return None
    colors = []
    for color_str in colors_str.split(';'):
        r, g, b = map(float, color_str.split(','))
        colors.append((r, g, b))
    return colors


def main():
    parser = argparse.ArgumentParser(description='Re-render B-spline strokes')
    parser.add_argument('strokes_path', type=str, help='Path to strokes.pkl file')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output PNG path')
    parser.add_argument('--indices', type=str, default=None, 
                        help='Comma-separated stroke indices to render (default: all)')
    parser.add_argument('--colors', type=str, default=None,
                        help='Colors for each stroke: "r,g,b;r,g,b;..." (0-1 range)')
    parser.add_argument('--phase-colors', action='store_true',
                        help='Color strokes by their phase assignment')
    parser.add_argument('--background', type=str, default='1.0,1.0,1.0',
                        help='Background color: "r,g,b" (default: white)')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Import here to avoid slow startup
    from src.painters.bspline_painter import BsplinePainter
    
    # Load strokes
    strokes_path = Path(args.strokes_path)
    if not strokes_path.exists():
        print(f"Error: {strokes_path} not found")
        sys.exit(1)
    
    device = torch.device(args.device)
    painter = BsplinePainter.load_strokes(strokes_path, device=device)
    
    # Parse indices
    indices = parse_indices(args.indices)
    if indices is None:
        indices = list(range(len(painter.shapes)))
    
    # Parse colors
    if args.phase_colors:
        # Color by phase
        cmap = cm.get_cmap('tab10', 10)
        colors = []
        for idx in indices:
            if painter.stroke_phase_index and idx < len(painter.stroke_phase_index):
                phase_idx = painter.stroke_phase_index[idx]
            else:
                phase_idx = 0
            rgba = cmap.colors[phase_idx]
            colors.append((float(rgba[0]), float(rgba[1]), float(rgba[2])))
    elif args.colors:
        colors = parse_colors(args.colors)
        # Extend colors if needed
        while len(colors) < len(indices):
            colors.append(colors[-1] if colors else (0.0, 0.0, 0.0))
    else:
        colors = [(0.0, 0.0, 0.0)] * len(indices)
    
    # Parse background
    bg_parts = args.background.split(',')
    background = (float(bg_parts[0]), float(bg_parts[1]), float(bg_parts[2]))
    
    # Render
    print(f"Rendering {len(indices)} strokes...")
    painter.render_to_png(
        args.output,
        stroke_indices=indices,
        colors=colors,
        background_color=background,
    )
    
    print(f"Done! Output saved to {args.output}")
    
    # Print stroke info
    print(f"\nStroke info:")
    print(f"  Total strokes: {len(painter.shapes)}")
    if painter.stroke_phase_index:
        for phase_idx in range(max(painter.stroke_phase_index) + 1):
            phase_strokes = [i for i, p in enumerate(painter.stroke_phase_index) if p == phase_idx]
            print(f"  Phase {phase_idx + 1}: strokes {phase_strokes}")


if __name__ == '__main__':
    main()

