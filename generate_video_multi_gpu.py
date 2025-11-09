#!/usr/bin/env python3
"""
Multi-GPU batch processing: Generate 30fps videos from 4 sparse frames using ViewCrafter.

Usage:
    # Process all scenes with all available GPUs
    python generate_video_multi_gpu.py --data_dir /path/to/data --output_dir /path/to/output

    # Use specific GPUs
    python generate_video_multi_gpu.py --data_dir /path/to/data --output_dir /path/to/output --gpus 0,1,2,3

    # Process a single range of scenes
    python generate_video_multi_gpu.py --data_dir /path/to/data --output_dir /path/to/output --ranges "100:200"

    # Process multiple ranges (use -1 for open-ended ranges)
    python generate_video_multi_gpu.py --data_dir /path/to/data --output_dir /path/to/output --ranges "100:200,250:-1" --gpus 0,1,2,3
"""

import sys
sys.path.append('./extern/dust3r')

import os
import torch
import glob
import time
import random
from pathlib import Path
import argparse
from datetime import datetime, timedelta
from multiprocessing import Process, Queue, Manager
from pytorch_lightning import seed_everything
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich import box

from dust3r.inference import load_model
from utils.pvd_utils import save_video
from video_generation_utils import load_diffusion_model, process_single_scene

console = Console()


def parse_ranges(ranges_str):
    """
    Parse range specification string into list of (start, end) tuples.

    Args:
        ranges_str: String like "100:200,250:-1" where -1 means open-ended

    Returns:
        List of (start, end) tuples. end=None means open-ended.

    Examples:
        "100:200" -> [(100, 200)]
        "100:200,250:-1" -> [(100, 200), (250, None)]
        "0:50,100:150,200:-1" -> [(0, 50), (100, 150), (200, None)]
    """
    if not ranges_str:
        return [(None, None)]

    ranges = []
    for range_part in ranges_str.split(','):
        range_part = range_part.strip()
        if ':' not in range_part:
            raise ValueError(f"Invalid range format: {range_part}. Expected 'start:end'")

        start_str, end_str = range_part.split(':', 1)
        start = int(start_str.strip()) if start_str.strip() else None
        end_val = end_str.strip()

        if end_val == '-1' or end_val == '':
            end = None
        else:
            end = int(end_val)

        ranges.append((start, end))

    return ranges


def in_ranges(scene_id, ranges):
    """
    Check if a scene_id falls within any of the specified ranges.

    Args:
        scene_id: Scene ID to check
        ranges: List of (start, end) tuples

    Returns:
        True if scene_id is in any range, False otherwise
    """
    if ranges == [(None, None)]:
        return True

    for start, end in ranges:
        # Check lower bound
        if start is not None and scene_id < start:
            continue
        # Check upper bound
        if end is not None and scene_id >= end:
            continue
        return True

    return False


def find_scenes(data_dir, ranges=None):
    """
    Find all scene directories in the data directory.

    Args:
        data_dir: Root data directory
        ranges: List of (start, end) tuples for filtering. None means all scenes.

    Returns:
        List of (scene_id, scene_path) tuples
    """
    if ranges is None:
        ranges = [(None, None)]

    all_scenes = []

    for item in sorted(os.listdir(data_dir)):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and item.isdigit():
            scene_id = int(item)

            # Check if scene has an images directory
            images_dir = os.path.join(item_path, 'images')
            if os.path.exists(images_dir):
                # Apply range filter
                if in_ranges(scene_id, ranges):
                    all_scenes.append((scene_id, item_path))

    return all_scenes


def find_pending_scenes(data_dir, output_dir, ranges=None):
    """
    Find scenes that haven't been processed yet (no mp4 file in output_dir).

    Args:
        data_dir: Root data directory
        output_dir: Output directory for videos
        ranges: List of (start, end) tuples for filtering

    Returns:
        List of scene_id integers for scenes that need processing
    """
    all_scenes = find_scenes(data_dir, ranges)
    pending = []

    for scene_id, scene_path in all_scenes:
        output_path = os.path.join(output_dir, f"{scene_id}.mp4")
        if not os.path.exists(output_path):
            pending.append(scene_id)

    return pending


def worker_process(gpu_id, task_queue, status_dict, data_dir, output_dir, args):
    """
    Worker process that processes scenes on a specific GPU.

    Args:
        gpu_id: GPU device ID
        task_queue: Queue of scene IDs to process
        status_dict: Shared dict for status reporting
        data_dir: Root data directory
        output_dir: Output directory for videos
        args: Arguments namespace
    """
    # Set GPU for this process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device('cuda:0')  # Always 0 since CUDA_VISIBLE_DEVICES is set

    # Random delay to avoid filesystem conflicts
    time.sleep(random.uniform(0, 1.0))

    # Initialize status for this GPU
    status_dict[gpu_id] = {
        'status': 'initializing',
        'scene': None,
        'completed': 0,
        'failed': 0,
        'start_time': None
    }

    try:
        # Load models
        seed_everything(args.seed)
        dust3r_model = load_model(args.dust3r_path, device)
        diffusion_model = load_diffusion_model(args.ckpt_path, args.config, device)

        # Update status to idle
        status_dict[gpu_id] = {
            'status': 'idle',
            'scene': None,
            'completed': 0,
            'failed': 0,
            'start_time': None
        }

        # Process scenes from queue
        while True:
            try:
                scene_id = task_queue.get(timeout=1)
            except:
                if task_queue.empty():
                    # Mark as finished
                    current = status_dict[gpu_id]
                    status_dict[gpu_id] = {
                        'status': 'finished',
                        'scene': None,
                        'completed': current['completed'],
                        'failed': current['failed'],
                        'start_time': None
                    }
                    break
                continue

            if scene_id is None:  # Poison pill
                current = status_dict[gpu_id]
                status_dict[gpu_id] = {
                    'status': 'finished',
                    'scene': None,
                    'completed': current['completed'],
                    'failed': current['failed'],
                    'start_time': None
                }
                break

            # Update status to processing
            current = status_dict[gpu_id]
            status_dict[gpu_id] = {
                'status': 'processing',
                'scene': scene_id,
                'completed': current['completed'],
                'failed': current['failed'],
                'start_time': time.time()
            }

            try:
                # Find image files
                scene_path = os.path.join(data_dir, str(scene_id))
                images_dir = os.path.join(scene_path, 'images')
                image_files = sorted(glob.glob(os.path.join(images_dir, "frame_*.jpg")))
                if len(image_files) == 0:
                    image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
                if len(image_files) == 0:
                    image_files = sorted(glob.glob(os.path.join(images_dir, "*.png")))

                if len(image_files) != 4:
                    raise ValueError(f"Expected 4 frames, found {len(image_files)}")

                # Process scene
                diffusion_results, stats = process_single_scene(
                    image_files, dust3r_model, diffusion_model, device,
                    num_frames=args.num_frames, ddim_steps=args.ddim_steps,
                    return_stats=True
                )

                # Save video
                output_path = os.path.join(output_dir, f"{scene_id}.mp4")
                save_video((diffusion_results + 1.0) / 2.0, output_path)

                # Update status - success
                current = status_dict[gpu_id]
                status_dict[gpu_id] = {
                    'status': 'idle',
                    'scene': None,
                    'completed': current['completed'] + 1,
                    'failed': current['failed'],
                    'start_time': None
                }

                # Force cleanup
                import gc
                torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                # Update status - failed
                current = status_dict[gpu_id]
                status_dict[gpu_id] = {
                    'status': 'idle',
                    'scene': None,
                    'completed': current['completed'],
                    'failed': current['failed'] + 1,
                    'start_time': None
                }

    except Exception as e:
        # Model loading or other critical error
        status_dict[gpu_id] = {
            'status': 'error',
            'scene': None,
            'completed': status_dict.get(gpu_id, {}).get('completed', 0),
            'failed': status_dict.get(gpu_id, {}).get('failed', 0),
            'start_time': None
        }


def format_time(seconds):
    """Format seconds into human-readable time."""
    if seconds is None:
        return "N/A"
    return str(timedelta(seconds=int(seconds)))


def create_status_display(status_dict, total_scenes, start_time, gpu_ids):
    """Create a rich layout for displaying multi-GPU status."""

    # Calculate overall statistics
    total_completed = sum(status_dict.get(gpu_id, {}).get('completed', 0) for gpu_id in gpu_ids)
    total_failed = sum(status_dict.get(gpu_id, {}).get('failed', 0) for gpu_id in gpu_ids)
    total_processed = total_completed + total_failed

    elapsed_time = time.time() - start_time

    # Calculate ETA
    if total_processed > 0:
        avg_time = elapsed_time / total_processed
        remaining = total_scenes - total_processed
        eta = avg_time * remaining
    else:
        eta = None

    # Create summary panel
    summary_text = Text()
    summary_text.append("Progress: ", style="bold")
    summary_text.append(f"{total_processed}/{total_scenes} ", style="bold cyan")
    summary_text.append("(", style="dim")
    summary_text.append(f"✓ {total_completed} ", style="bold green")
    summary_text.append(f"✗ {total_failed}", style="bold red")
    summary_text.append(")", style="dim")

    summary_text.append(" │ ", style="dim")
    summary_text.append("Elapsed: ", style="bold")
    summary_text.append(format_time(elapsed_time), style="yellow")

    if eta is not None:
        summary_text.append(" │ ", style="dim")
        summary_text.append("ETA: ", style="bold")
        summary_text.append(format_time(eta), style="yellow")

    summary_panel = Panel(
        summary_text,
        title="[bold magenta]Overall Progress[/bold magenta]",
        border_style="cyan",
        box=box.ROUNDED
    )

    # Create GPU status table
    gpu_table = Table(
        show_header=True,
        header_style="bold magenta",
        box=box.ROUNDED,
        padding=(0, 1)
    )
    gpu_table.add_column("GPU", style="cyan", width=6)
    gpu_table.add_column("Status", width=12)
    gpu_table.add_column("Current Scene", width=15)
    gpu_table.add_column("Completed", justify="right", width=10)
    gpu_table.add_column("Failed", justify="right", width=8)
    gpu_table.add_column("Processing Time", justify="right", width=16)

    for gpu_id in gpu_ids:
        info = status_dict.get(gpu_id, {})
        status = info.get('status', 'unknown')
        scene = info.get('scene')
        completed = info.get('completed', 0)
        failed = info.get('failed', 0)
        start_time_gpu = info.get('start_time')

        # Format status with colors
        if status == 'processing':
            status_str = "[bold yellow]Processing[/bold yellow]"
        elif status == 'idle':
            status_str = "[green]Idle[/green]"
        elif status == 'initializing':
            status_str = "[blue]Loading...[/blue]"
        elif status == 'finished':
            status_str = "[bold green]Finished[/bold green]"
        elif status == 'error':
            status_str = "[bold red]Error[/bold red]"
        else:
            status_str = "[dim]Unknown[/dim]"

        # Format scene
        scene_str = str(scene) if scene is not None else "-"

        # Format processing time
        if start_time_gpu is not None:
            proc_time = time.time() - start_time_gpu
            time_str = f"{proc_time:.1f}s"
        else:
            time_str = "-"

        gpu_table.add_row(
            f"GPU {gpu_id}",
            status_str,
            scene_str,
            f"[green]{completed}[/green]",
            f"[red]{failed}[/red]" if failed > 0 else "0",
            time_str
        )

    gpu_panel = Panel(
        gpu_table,
        title="[bold magenta]GPU Status[/bold magenta]",
        border_style="cyan",
        box=box.ROUNDED
    )

    # Create layout
    layout = Layout()
    layout.split_column(
        Layout(summary_panel, size=3),
        Layout(gpu_panel)
    )

    return layout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory containing scene subdirectories')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for videos')
    parser.add_argument('--ranges', type=str, default=None, help='Comma-separated ranges (e.g., "100:200,250:-1"). Use -1 for open-ended. If not specified, processes all scenes.')
    parser.add_argument('--gpus', type=str, default=None, help='Comma-separated GPU IDs (e.g., "0,1,2,3"). If not specified, uses all available GPUs.')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/model_sparse.ckpt')
    parser.add_argument('--dust3r_path', type=str, default='./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth')
    parser.add_argument('--config', type=str, default='./configs/inference_pvd_1024.yaml')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--ddim_steps', type=int, default=50)
    parser.add_argument('--num_frames', type=int, default=30)
    args = parser.parse_args()

    # Determine GPU IDs
    if args.gpus:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    else:
        # Use all available GPUs
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            console.print("[red]No GPUs available![/red]")
            return
        gpu_ids = list(range(gpu_count))

    # Parse ranges
    ranges = parse_ranges(args.ranges) if args.ranges else None

    # Print header
    console.print("\n[bold magenta]╔═══════════════════════════════════════════════════════════════╗[/bold magenta]")
    console.print("[bold magenta]║[/bold magenta]     [bold cyan]ViewCrafter Multi-GPU Batch Video Generation[/bold cyan]        [bold magenta]║[/bold magenta]")
    console.print("[bold magenta]╚═══════════════════════════════════════════════════════════════╝[/bold magenta]\n")

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Find pending scenes
    console.print("[bold cyan]Scanning for pending scenes...[/bold cyan]")
    pending_scenes = find_pending_scenes(args.data_dir, args.output_dir, ranges=ranges)

    if not pending_scenes:
        console.print(f"[green]No pending scenes found. All scenes already processed![/green]")
        return

    # Print info
    console.print(f"[bold]Found {len(pending_scenes)} pending scenes[/bold]")
    if ranges and ranges != [(None, None)]:
        ranges_str = ", ".join([f"[{s if s is not None else '0'}:{e if e is not None else '∞'})" for s, e in ranges])
        console.print(f"Range filter: {ranges_str}")
    console.print(f"Data directory: [cyan]{args.data_dir}[/cyan]")
    console.print(f"Output directory: [cyan]{args.output_dir}[/cyan]")
    console.print(f"Using GPUs: [cyan]{', '.join(map(str, gpu_ids))}[/cyan]\n")

    # Create shared structures
    manager = Manager()
    task_queue = Queue()
    status_dict = manager.dict()

    # Populate task queue
    for scene_id in pending_scenes:
        task_queue.put(scene_id)

    # Add poison pills
    for _ in gpu_ids:
        task_queue.put(None)

    # Start worker processes
    console.print(f"[bold cyan]Starting {len(gpu_ids)} worker processes...[/bold cyan]\n")
    workers = []
    for gpu_id in gpu_ids:
        p = Process(
            target=worker_process,
            args=(gpu_id, task_queue, status_dict, args.data_dir, args.output_dir, args)
        )
        p.start()
        workers.append(p)

    # Monitor progress with live display
    start_time = time.time()

    with Live(
        create_status_display(status_dict, len(pending_scenes), start_time, gpu_ids),
        refresh_per_second=2,
        console=console
    ) as live:
        while any(p.is_alive() for p in workers):
            live.update(create_status_display(status_dict, len(pending_scenes), start_time, gpu_ids))
            time.sleep(0.5)

    # Wait for all workers to complete
    for p in workers:
        p.join()

    # Final summary
    total_time = time.time() - start_time
    total_completed = sum(status_dict.get(gpu_id, {}).get('completed', 0) for gpu_id in gpu_ids)
    total_failed = sum(status_dict.get(gpu_id, {}).get('failed', 0) for gpu_id in gpu_ids)

    console.print("\n")
    summary_table = Table(
        title="[bold magenta]Final Summary[/bold magenta]",
        box=box.DOUBLE,
        show_header=True,
        header_style="bold magenta"
    )
    summary_table.add_column("Metric", style="cyan", no_wrap=True)
    summary_table.add_column("Value", justify="right", style="green")

    summary_table.add_row("Total scenes", str(len(pending_scenes)))
    summary_table.add_row("Successful", f"[green]{total_completed}[/green]")
    summary_table.add_row("Failed", f"[red]{total_failed}[/red]" if total_failed > 0 else "0")
    summary_table.add_row("Success rate", f"{total_completed/(total_completed+total_failed)*100:.1f}%" if (total_completed+total_failed) > 0 else "N/A")
    summary_table.add_row("", "")
    summary_table.add_row("GPUs used", str(len(gpu_ids)))
    summary_table.add_row("Total time", format_time(total_time))
    summary_table.add_row("Avg per scene", f"{total_time/(total_completed+total_failed):.1f}s" if (total_completed+total_failed) > 0 else "N/A")

    console.print(summary_table)
    console.print(f"\n[bold green]✓ Multi-GPU batch processing completed![/bold green]")
    console.print(f"[cyan]Videos saved to: {args.output_dir}[/cyan]\n")


if __name__ == "__main__":
    main()
