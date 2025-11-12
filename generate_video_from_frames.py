#!/usr/bin/env python3
"""
Single scene: Generate 30fps video from 4 sparse frames using ViewCrafter.

Usage:
    python generate_video_from_frames.py --image_dir /path/to/images --output_path /path/to/output.mp4
"""

import sys
sys.path.append('./extern/dust3r')

import os
import torch
import glob
import time
from pathlib import Path
import argparse
from pytorch_lightning import seed_everything
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from dust3r.inference import load_model
from utils.pvd_utils import save_video
from video_generation_utils import load_diffusion_model, process_single_scene

console = Console()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing frame_*.jpg')
    parser.add_argument('--output_path', type=str, required=True, help='Output video path (e.g., output.mp4)')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/model_sparse.ckpt')
    parser.add_argument('--dust3r_path', type=str, default='./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth')
    parser.add_argument('--config', type=str, default='./configs/inference_pvd_1024.yaml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--ddim_steps', type=int, default=50)
    parser.add_argument('--num_frames', type=int, default=30)
    args = parser.parse_args()

    start_total = time.time()

    seed_everything(args.seed)
    device = torch.device(args.device)

    # Find frames
    image_files = sorted(glob.glob(os.path.join(args.image_dir, "frame_*.jpg")))
    if len(image_files) == 0:
        image_files = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")))
    if len(image_files) == 0:
        image_files = sorted(glob.glob(os.path.join(args.image_dir, "*.png")))

    if len(image_files) != 4:
        console.print(f"[red]Error: Expected 4 frames, found {len(image_files)}[/red]")
        return

    # Load models
    console.print("\n[bold cyan]Loading models...[/bold cyan]")
    start_load = time.time()
    dust3r_model = load_model(args.dust3r_path, device)
    diffusion_model = load_diffusion_model(args.ckpt_path, args.config, device)
    load_time = time.time() - start_load
    console.print(f"[green]✓ Models loaded in {load_time:.2f}s[/green]")

    # Process scene
    console.print(f"\n[bold cyan]Processing {args.image_dir}...[/bold cyan]")
    start_process = time.time()
    diffusion_results, stats = process_single_scene(
        image_files, dust3r_model, diffusion_model, device,
        num_frames=args.num_frames, ddim_steps=args.ddim_steps,
        return_stats=True
    )
    process_time = time.time() - start_process

    # Save video
    console.print(f"\n[bold cyan]Saving video...[/bold cyan]")
    start_save = time.time()
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    save_video((diffusion_results + 1.0) / 2.0, args.output_path, fps=30)
    save_time = time.time() - start_save

    total_time = time.time() - start_total

    # Create timing summary table
    table = Table(title="Timing Summary", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("Stage", style="cyan", no_wrap=True)
    table.add_column("Time (s)", justify="right", style="green")
    table.add_column("Percentage", justify="right", style="yellow")

    table.add_row("Model loading", f"{load_time:.2f}", f"{load_time/total_time*100:.1f}%")
    table.add_row("Scene processing", f"{process_time:.2f}", f"{process_time/total_time*100:.1f}%")
    table.add_row("  ├─ Load images", f"{stats['load_images']:.2f}", f"{stats['load_images']/total_time*100:.1f}%")
    table.add_row("  ├─ DUSt3R reconstruction", f"{stats['dust3r_reconstruction']:.2f}", f"{stats['dust3r_reconstruction']/total_time*100:.1f}%")
    table.add_row("  ├─ Extract geometry", f"{stats['extract_geometry']:.2f}", f"{stats['extract_geometry']/total_time*100:.1f}%")
    table.add_row("  ├─ Render point cloud", f"{stats['render_pointcloud']:.2f}", f"{stats['render_pointcloud']/total_time*100:.1f}%")
    table.add_row("  └─ Diffusion refinement", f"{stats['diffusion_refinement']:.2f}", f"{stats['diffusion_refinement']/total_time*100:.1f}%")
    table.add_row("Video saving", f"{save_time:.2f}", f"{save_time/total_time*100:.1f}%")
    table.add_row("", "", "", style="dim")
    table.add_row("[bold]Total time[/bold]", f"[bold]{total_time:.2f}[/bold]", "[bold]100.0%[/bold]")

    console.print("\n")
    console.print(table)
    console.print(f"\n[bold green]✓ Saved to {args.output_path}[/bold green]\n")


if __name__ == "__main__":
    main()
