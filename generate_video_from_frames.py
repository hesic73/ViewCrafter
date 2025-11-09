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
from pathlib import Path
import argparse
from pytorch_lightning import seed_everything

from dust3r.inference import load_model
from utils.pvd_utils import save_video
from video_generation_utils import load_diffusion_model, process_single_scene


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
    
    seed_everything(args.seed)
    device = torch.device(args.device)
    
    # Find frames
    image_files = sorted(glob.glob(os.path.join(args.image_dir, "frame_*.jpg")))
    if len(image_files) == 0:
        image_files = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")))
    if len(image_files) == 0:
        image_files = sorted(glob.glob(os.path.join(args.image_dir, "*.png")))
    
    if len(image_files) != 4:
        print(f"Error: Expected 4 frames, found {len(image_files)}")
        return
    
    print(f"Loading models...")
    dust3r_model = load_model(args.dust3r_path, device)
    diffusion_model = load_diffusion_model(args.ckpt_path, args.config, device)
    
    print(f"Processing {args.image_dir}...")
    
    # Process scene
    diffusion_results = process_single_scene(
        image_files, dust3r_model, diffusion_model, device,
        num_frames=args.num_frames, ddim_steps=args.ddim_steps
    )
    
    # Save video
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    save_video((diffusion_results + 1.0) / 2.0, args.output_path)
    
    print(f"✓ Saved to {args.output_path}")


if __name__ == "__main__":
    main()
