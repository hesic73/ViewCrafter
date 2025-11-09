#!/usr/bin/env python3
"""
Batch processing: Generate videos for multiple scenes.

Directory structure:
  input_dir/
    ├── 21000/
    │   └── images/
    │       ├── frame_000.jpg
    │       ├── frame_001.jpg
    │       ├── frame_002.jpg
    │       └── frame_003.jpg
    ├── 21001/
    │   └── images/
    │       └── ...
    └── ...

Usage:
    python batch_generate_videos.py --input_dir /path/to/scenes --output_dir /path/to/videos
"""

import sys
sys.path.append('./extern/dust3r')

import os
import torch
import glob
from pathlib import Path
import argparse
from tqdm import tqdm
from pytorch_lightning import seed_everything

from dust3r.inference import load_model
from utils.pvd_utils import save_video
from video_generation_utils import load_diffusion_model, process_single_scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, 
                       help='Directory containing scene folders (e.g., 21000, 21001, ...)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for videos')
    parser.add_argument('--images_subdir', type=str, default='images',
                       help='Subdirectory name containing frames (default: images)')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/model_sparse.ckpt')
    parser.add_argument('--dust3r_path', type=str, default='./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth')
    parser.add_argument('--config', type=str, default='./configs/inference_pvd_1024.yaml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--ddim_steps', type=int, default=50)
    parser.add_argument('--num_frames', type=int, default=30)
    parser.add_argument('--verbose', action='store_true', help='Print detailed errors')
    args = parser.parse_args()
    
    seed_everything(args.seed)
    device = torch.device(args.device)
    
    # Find all scene directories
    input_path = Path(args.input_dir)
    scene_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])
    
    if len(scene_dirs) == 0:
        print(f"No scene directories found in {args.input_dir}")
        return
    
    print(f"Found {len(scene_dirs)} scenes to process")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print("="*60)
    
    # Load models once
    print("Loading DUSt3R model...")
    dust3r_model = load_model(args.dust3r_path, device)
    
    print("Loading diffusion model...")
    diffusion_model = load_diffusion_model(args.ckpt_path, args.config, device)
    
    print("="*60)
    print()
    
    # Process each scene
    successful = 0
    failed = 0
    
    for scene_dir in tqdm(scene_dirs, desc="Processing scenes"):
        scene_id = scene_dir.name
        image_dir = scene_dir / args.images_subdir
        output_path = Path(args.output_dir) / f"{scene_id}.mp4"
        
        # Skip if output already exists
        if output_path.exists() and not args.verbose:
            print(f"  ⊙ {scene_id}: Already exists, skipping")
            continue
        
        # Skip if images directory doesn't exist
        if not image_dir.exists():
            print(f"  ✗ {scene_id}: Images directory not found at {image_dir}")
            failed += 1
            continue
        
        # Find frames
        image_files = sorted(glob.glob(os.path.join(image_dir, "frame_*.jpg")))
        if len(image_files) == 0:
            image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        if len(image_files) == 0:
            image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        
        if len(image_files) != 4:
            print(f"  ✗ {scene_id}: Expected 4 frames, found {len(image_files)}")
            failed += 1
            continue
        
        try:
            # Process scene
            diffusion_results = process_single_scene(
                image_files, dust3r_model, diffusion_model, device,
                num_frames=args.num_frames, ddim_steps=args.ddim_steps
            )
            
            # Save video
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_video((diffusion_results + 1.0) / 2.0, str(output_path))
            
            successful += 1
            print(f"  ✓ {scene_id}")
            
        except Exception as e:
            failed += 1
            print(f"  ✗ {scene_id}: Error - {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # Summary
    print()
    print("="*60)
    print(f"Batch processing complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(scene_dirs)}")
    print("="*60)


if __name__ == "__main__":
    main()
