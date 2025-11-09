#!/usr/bin/env python3
"""
Batch processing: Generate 30fps videos from 4 sparse frames using ViewCrafter.

Usage:
    # Process all scenes
    python generate_video_batch.py --data_dir /path/to/data --output_dir /path/to/output

    # Process a range of scenes (for distributed processing)
    python generate_video_batch.py --data_dir /path/to/data --output_dir /path/to/output --start 21000 --end 21200
"""

import sys
sys.path.append('./extern/dust3r')

import os
import torch
import glob
import time
from pathlib import Path
import argparse
from datetime import datetime, timedelta
from pytorch_lightning import seed_everything
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.live import Live
from rich.layout import Layout
from rich import box

from dust3r.inference import load_model
from utils.pvd_utils import save_video
from video_generation_utils import load_diffusion_model, process_single_scene

console = Console()


def find_scenes(data_dir, start=None, end=None):
    """
    Find all scene directories in the data directory.

    Args:
        data_dir: Root data directory
        start: Start scene ID (inclusive)
        end: End scene ID (exclusive)

    Returns:
        List of (scene_id, scene_path) tuples
    """
    all_scenes = []

    # Find all subdirectories that look like scene IDs (numeric names)
    for item in sorted(os.listdir(data_dir)):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and item.isdigit():
            scene_id = int(item)

            # Check if scene has an images directory
            images_dir = os.path.join(item_path, 'images')
            if os.path.exists(images_dir):
                # Apply range filter if specified
                if start is not None and scene_id < start:
                    continue
                if end is not None and scene_id >= end:
                    continue

                all_scenes.append((scene_id, item_path))

    return all_scenes


def format_time(seconds):
    """Format seconds into human-readable time."""
    return str(timedelta(seconds=int(seconds)))


def create_stats_table(stats, total_time):
    """Create a rich table for timing statistics."""
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column("Stage", style="cyan")
    table.add_column("Time", justify="right", style="green")
    table.add_column("%", justify="right", style="yellow")

    table.add_row("Load images", f"{stats['load_images']:.1f}s", f"{stats['load_images']/total_time*100:.1f}%")
    table.add_row("DUSt3R recon", f"{stats['dust3r_reconstruction']:.1f}s", f"{stats['dust3r_reconstruction']/total_time*100:.1f}%")
    table.add_row("Extract geom", f"{stats['extract_geometry']:.1f}s", f"{stats['extract_geometry']/total_time*100:.1f}%")
    table.add_row("Render PCD", f"{stats['render_pointcloud']:.1f}s", f"{stats['render_pointcloud']/total_time*100:.1f}%")
    table.add_row("Diffusion", f"{stats['diffusion_refinement']:.1f}s", f"{stats['diffusion_refinement']/total_time*100:.1f}%")

    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory containing scene subdirectories')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for videos')
    parser.add_argument('--start', type=int, default=None, help='Start scene ID (inclusive)')
    parser.add_argument('--end', type=int, default=None, help='End scene ID (exclusive)')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/model_sparse.ckpt')
    parser.add_argument('--dust3r_path', type=str, default='./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth')
    parser.add_argument('--config', type=str, default='./configs/inference_pvd_1024.yaml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--ddim_steps', type=int, default=50)
    parser.add_argument('--num_frames', type=int, default=30)
    args = parser.parse_args()

    # Print header
    console.print("\n[bold magenta]╔═══════════════════════════════════════════════════════════════╗[/bold magenta]")
    console.print("[bold magenta]║[/bold magenta]          [bold cyan]ViewCrafter Batch Video Generation[/bold cyan]              [bold magenta]║[/bold magenta]")
    console.print("[bold magenta]╚═══════════════════════════════════════════════════════════════╝[/bold magenta]\n")

    seed_everything(args.seed)
    device = torch.device(args.device)

    # Find scenes
    scenes = find_scenes(args.data_dir, start=args.start, end=args.end)

    if not scenes:
        console.print(f"[red]No scenes found in {args.data_dir}[/red]")
        if args.start or args.end:
            console.print(f"[yellow]Range filter: {args.start or 'start'} to {args.end or 'end'}[/yellow]")
        return

    # Print range info
    range_str = f"Processing {len(scenes)} scenes"
    if args.start or args.end:
        range_str += f" (IDs: {args.start or scenes[0][0]} to {args.end or scenes[-1][0]+1})"
    console.print(f"[bold]{range_str}[/bold]")
    console.print(f"Data directory: [cyan]{args.data_dir}[/cyan]")
    console.print(f"Output directory: [cyan]{args.output_dir}[/cyan]\n")

    # Load models
    console.print("[bold cyan]Loading models...[/bold cyan]")
    start_load = time.time()
    dust3r_model = load_model(args.dust3r_path, device)
    diffusion_model = load_diffusion_model(args.ckpt_path, args.config, device)
    load_time = time.time() - start_load
    console.print(f"[green]✓ Models loaded in {load_time:.2f}s[/green]\n")

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Process scenes with rich progress display
    batch_start_time = time.time()
    successful = 0
    failed = 0
    all_stats = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:

        task = progress.add_task("[cyan]Processing scenes...", total=len(scenes))

        for i, (scene_id, scene_path) in enumerate(scenes):
            progress.update(task, description=f"[cyan]Processing scene {scene_id}")

            # Find image files
            images_dir = os.path.join(scene_path, 'images')
            image_files = sorted(glob.glob(os.path.join(images_dir, "frame_*.jpg")))
            if len(image_files) == 0:
                image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
            if len(image_files) == 0:
                image_files = sorted(glob.glob(os.path.join(images_dir, "*.png")))

            if len(image_files) != 4:
                console.print(f"[yellow]⚠ Scene {scene_id}: Expected 4 frames, found {len(image_files)}. Skipping.[/yellow]")
                failed += 1
                progress.update(task, advance=1)
                continue

            try:
                # Process scene
                scene_start = time.time()
                diffusion_results, stats = process_single_scene(
                    image_files, dust3r_model, diffusion_model, device,
                    num_frames=args.num_frames, ddim_steps=args.ddim_steps,
                    return_stats=True
                )
                scene_time = time.time() - scene_start

                # Save video
                output_path = os.path.join(args.output_dir, f"{scene_id}.mp4")
                save_video((diffusion_results + 1.0) / 2.0, output_path)

                all_stats.append(stats)
                successful += 1

                # Calculate ETA
                avg_time = (time.time() - batch_start_time) / (i + 1)
                remaining = len(scenes) - (i + 1)
                eta = avg_time * remaining

                # Create mini panel for this scene
                stats_table = create_stats_table(stats, scene_time)
                panel_content = f"[green]✓[/green] Scene {scene_id} completed in [bold]{scene_time:.1f}s[/bold]\n\n{stats_table}\n\nETA: {format_time(eta)}"
                console.print(Panel(panel_content, title=f"Scene {scene_id}", border_style="green", box=box.ROUNDED))

            except Exception as e:
                console.print(f"[red]✗ Scene {scene_id} failed: {str(e)}[/red]")
                failed += 1

            progress.update(task, advance=1)

    # Final summary
    total_batch_time = time.time() - batch_start_time

    # Create summary table
    summary_table = Table(title="Batch Processing Summary", box=box.DOUBLE, show_header=True, header_style="bold magenta")
    summary_table.add_column("Metric", style="cyan", no_wrap=True)
    summary_table.add_column("Value", justify="right", style="green")

    summary_table.add_row("Total scenes", str(len(scenes)))
    summary_table.add_row("Successful", f"[green]{successful}[/green]")
    summary_table.add_row("Failed", f"[red]{failed}[/red]" if failed > 0 else "0")
    summary_table.add_row("Success rate", f"{successful/len(scenes)*100:.1f}%")
    summary_table.add_row("", "")
    summary_table.add_row("Total time", format_time(total_batch_time))
    summary_table.add_row("Avg per scene", f"{total_batch_time/len(scenes):.1f}s")
    summary_table.add_row("Model loading", f"{load_time:.1f}s")

    if all_stats:
        avg_stats = {
            key: sum(s[key] for s in all_stats) / len(all_stats)
            for key in all_stats[0].keys()
        }
        summary_table.add_row("", "")
        summary_table.add_row("[bold]Avg pipeline breakdown:[/bold]", "")
        summary_table.add_row("  Load images", f"{avg_stats['load_images']:.1f}s")
        summary_table.add_row("  DUSt3R reconstruction", f"{avg_stats['dust3r_reconstruction']:.1f}s")
        summary_table.add_row("  Extract geometry", f"{avg_stats['extract_geometry']:.1f}s")
        summary_table.add_row("  Render point cloud", f"{avg_stats['render_pointcloud']:.1f}s")
        summary_table.add_row("  Diffusion refinement", f"{avg_stats['diffusion_refinement']:.1f}s")

    console.print("\n")
    console.print(summary_table)
    console.print(f"\n[bold green]✓ Batch processing completed![/bold green]")
    console.print(f"[cyan]Videos saved to: {args.output_dir}[/cyan]\n")


if __name__ == "__main__":
    main()
