#!/usr/bin/env python3
"""
Shared utilities for video generation from sparse frames.
"""

import sys
sys.path.append('./extern/dust3r')

import time
import torch
import numpy as np
import torch.nn.functional as F
from scipy.spatial.transform import Rotation, Slerp

from dust3r.inference import inference, load_model
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import to_numpy

from pytorch3d.renderer import (
    PointsRasterizationSettings, PointsRenderer, PointsRasterizer,
    AlphaCompositor, PerspectiveCameras
)
from pytorch3d.structures import Pointclouds

from utils.diffusion_utils import instantiate_from_config, load_model_checkpoint, image_guided_synthesis
from utils.pvd_utils import save_video
from omegaconf import OmegaConf


def catmull_rom_interp(points, samples_per_seg=20):
    """
    Catmull-Rom spline interpolation for smooth camera trajectories.
    
    Args:
        points: Control points [N, D]
        samples_per_seg: Number of samples per segment
    
    Returns:
        dense_points: Densely sampled points
        dense_times: Time values for dense points
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return pts, np.arange(n, dtype=np.float64)
    
    extended = np.vstack([pts[0:1], pts, pts[-1:], pts[-1:]])
    dense_len = (n - 1) * samples_per_seg + 1
    dense_times = np.linspace(0, n - 1, dense_len)
    
    result = []
    for u in dense_times:
        if u >= n - 1:
            result.append(pts[-1])
            continue
        
        i = int(np.floor(u))
        t = u - i
        p0, p1, p2, p3 = extended[i], extended[i+1], extended[i+2], extended[i+3]
        
        t2, t3 = t * t, t * t * t
        pos = 0.5 * ((2*p1) + (-p0+p2)*t + (2*p0-5*p1+4*p2-p3)*t2 + (-p0+3*p1-3*p2+p3)*t3)
        result.append(pos)
    
    return np.asarray(result), dense_times


def interpolate_cameras(c2ws, focals, pps, num_frames, device):
    """
    Interpolate camera poses using Catmull-Rom (positions) + SLERP (rotations).

    Args:
        c2ws: Camera-to-world matrices [N, 3, 4] or [N, 4, 4]
        focals: Focal lengths [N]
        pps: Principal points [N, 2]
        num_frames: Target number of frames
        device: torch device

    Returns:
        interp_c2ws: Interpolated camera-to-world matrices [num_frames, 3, 4]
        interp_focals: Interpolated focal lengths [num_frames]
        interp_pps: Interpolated principal points [num_frames, 2]
    """
    n = c2ws.shape[0]
    c2ws_np = c2ws.cpu().numpy()

    # Convert to 4x4 (handle both [N, 3, 4] and [N, 4, 4] inputs)
    c2ws_4x4 = []
    for i in range(n):
        if c2ws_np[i].shape == (4, 4):
            # Already 4x4
            c2ws_4x4.append(c2ws_np[i])
        else:
            # Convert 3x4 to 4x4
            mat = np.eye(4)
            mat[:3, :] = c2ws_np[i]
            c2ws_4x4.append(mat)
    
    # Extract positions and rotations
    positions = np.array([mat[:3, 3] for mat in c2ws_4x4])
    rotations = [Rotation.from_matrix(mat[:3, :3]) for mat in c2ws_4x4]
    
    # Dense Catmull-Rom for positions
    trans_dense, dense_times = catmull_rom_interp(positions, samples_per_seg=20)
    
    # Dense SLERP for rotations
    quats = np.array([r.as_quat() for r in rotations])
    for i in range(1, len(quats)):
        if np.dot(quats[i-1], quats[i]) < 0:
            quats[i] = -quats[i]
    
    key_times = np.arange(len(quats), dtype=np.float64)
    rot_dense = Slerp(key_times, Rotation.from_quat(quats))(dense_times)
    
    # Resample to target frames
    target_times = np.linspace(0, dense_times[-1], num_frames)
    interp_positions = np.array([np.interp(target_times, dense_times, trans_dense[:, i]) for i in range(3)]).T
    interp_rotations = Slerp(dense_times, rot_dense)(target_times)
    
    # Build interpolated c2ws
    interp_c2ws = []
    rot_matrices = interp_rotations.as_matrix()
    for i in range(num_frames):
        mat = np.eye(4)
        mat[:3, :3] = rot_matrices[i]
        mat[:3, 3] = interp_positions[i]
        interp_c2ws.append(mat[:3, :])
    
    interp_c2ws = torch.from_numpy(np.stack(interp_c2ws)).float().to(device)
    
    # Interpolate intrinsics
    focals_np = focals.cpu().numpy().flatten()
    pps_np = pps.cpu().numpy()
    key_indices = np.linspace(0, num_frames-1, n)
    target_indices = np.arange(num_frames)
    
    interp_focals = np.interp(target_indices, key_indices, focals_np)
    interp_pps = np.stack([
        np.interp(target_indices, key_indices, pps_np[:, 0]),
        np.interp(target_indices, key_indices, pps_np[:, 1])
    ], axis=1)
    
    interp_focals = torch.from_numpy(interp_focals).float().to(device)
    interp_pps = torch.from_numpy(interp_pps).float().to(device)
    
    return interp_c2ws, interp_focals, interp_pps


def setup_renderer(c2ws, focals, pps, H, W, device):
    """
    Setup PyTorch3D point cloud renderer.
    
    Args:
        c2ws: Camera-to-world matrices [N, 3, 4]
        focals: Focal lengths [N]
        pps: Principal points [N, 2]
        H, W: Image height and width
        device: torch device
    
    Returns:
        renderer: PyTorch3D PointsRenderer
    """
    R, T = c2ws[:,:3, :3], c2ws[:,:3, 3:]
    R = torch.stack([-R[:,:, 0], -R[:,:, 1], R[:,:, 2]], 2)
    new_c2w = torch.cat([R, T], 2)
    w2c = torch.linalg.inv(torch.cat((new_c2w, torch.tensor([[[0,0,0,1]]]).to(device).repeat(new_c2w.shape[0],1,1)),1))
    R_new, T_new = w2c[:,:3, :3].permute(0,2,1), w2c[:,:3, 3]
    
    cameras = PerspectiveCameras(
        focal_length=focals.unsqueeze(-1), 
        principal_point=pps,
        in_ndc=False, 
        image_size=[(H, W)],
        R=R_new, 
        T=T_new, 
        device=device
    )
    
    raster_settings = PointsRasterizationSettings(
        image_size=(H, W), radius=0.01, points_per_pixel=10, bin_size=0
    )
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
        compositor=AlphaCompositor()
    )
    return renderer


def render_pcd(pts3d, imgs, masks, renderer, device):
    """
    Render point cloud from given viewpoints.

    Args:
        pts3d: List of 3D points [N, H, W, 3]
        imgs: List of RGB images [N, H, W, 3]
        masks: List of masks [N, H, W] or None
        renderer: PyTorch3D PointsRenderer
        device: torch device

    Returns:
        images: Rendered images [num_views, H, W, 4]
    """
    imgs_np = to_numpy(imgs)
    pts3d_np = to_numpy(pts3d)

    if masks is None:
        pts = torch.from_numpy(np.concatenate([p for p in pts3d_np])).view(-1, 3).to(device)
        col = torch.from_numpy(np.concatenate([p for p in imgs_np])).view(-1, 3).to(device)
    else:
        masks_np = to_numpy(masks)
        pts = torch.from_numpy(np.concatenate([p[m] for p, m in zip(pts3d_np, masks_np)])).to(device)
        col = torch.from_numpy(np.concatenate([p[m] for p, m in zip(imgs_np, masks_np)])).to(device)

    # Handle PyTorch3D API change: _cameras -> rasterizer.cameras
    if hasattr(renderer, '_cameras'):
        num_cameras = len(renderer._cameras)
    else:
        num_cameras = len(renderer.rasterizer.cameras)

    point_cloud = Pointclouds(points=[pts], features=[col]).extend(num_cameras)
    images = renderer(point_cloud)
    return images


def load_diffusion_model(ckpt_path, config_path, device):
    """
    Load ViewCrafter diffusion model.
    
    Args:
        ckpt_path: Path to model checkpoint
        config_path: Path to config yaml
        device: torch device
    
    Returns:
        model: Loaded diffusion model
    """
    config = OmegaConf.load(config_path)
    model_config = config.pop("model", OmegaConf.create())
    model_config['params']['unet_config']['params']['use_checkpoint'] = False
    model = instantiate_from_config(model_config).to(device)
    model.perframe_ae = True
    model = load_model_checkpoint(model, ckpt_path)
    model.eval()
    return model


def run_diffusion_chunks(render_results, diffusion_model, num_frames, ddim_steps, device):
    """
    Run diffusion refinement in chunks.
    
    Args:
        render_results: Rendered frames [num_frames, H, W, 3]
        diffusion_model: ViewCrafter model
        num_frames: Number of output frames
        ddim_steps: Number of DDIM steps
        device: torch device
    
    Returns:
        diffusion_results: Refined frames [num_frames, H, W, 3]
    """
    chunk_size = 25
    diffusion_results = []
    
    num_chunks = (num_frames + chunk_size - 2) // (chunk_size - 1)
    for i in range(num_chunks):
        start_idx = i * (chunk_size - 1)
        end_idx = min(start_idx + chunk_size, num_frames)
        
        chunk = render_results[start_idx:end_idx]
        if len(chunk) < chunk_size:
            padding = chunk[-1:].repeat(chunk_size - len(chunk), 1, 1, 1)
            chunk = torch.cat([chunk, padding], dim=0)
        
        # Run diffusion
        videos = (chunk * 2. - 1.).permute(3,0,1,2).unsqueeze(0).to(device)
        h, w = 576 // 8, 1024 // 8
        noise_shape = [1, diffusion_model.model.diffusion_model.out_channels, chunk_size, h, w]
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            batch_samples = image_guided_synthesis(
                diffusion_model, ['Rotating view of a scene'], videos, noise_shape,
                n_samples=1, ddim_steps=ddim_steps, ddim_eta=1.0,
                unconditional_guidance_scale=7.5, cfg_img=None, fs=10,
                text_input=True, multiple_cond_cfg=False, timestep_spacing='uniform_trailing',
                guidance_rescale=0.7, condition_index=[0]
            )
        
        result = torch.clamp(batch_samples[0][0].permute(1,2,3,0), -1., 1.)
        
        if i == 0:
            diffusion_results.append(result)
        else:
            diffusion_results.append(result[1:])
    
    diffusion_results = torch.cat(diffusion_results, dim=0)[:num_frames]
    return diffusion_results


def process_single_scene(image_files, dust3r_model, diffusion_model, device, num_frames=30, ddim_steps=50, return_stats=False):
    """
    Complete pipeline to process a single scene.

    Args:
        image_files: List of 4 image file paths
        dust3r_model: DUSt3R model
        diffusion_model: ViewCrafter model
        device: torch device
        num_frames: Number of output frames
        ddim_steps: Number of DDIM steps
        return_stats: If True, return (results, stats) tuple. Otherwise just return results.

    Returns:
        If return_stats=False: diffusion_results (Final video frames [num_frames, H, W, 3] in range [-1, 1])
        If return_stats=True: (diffusion_results, stats_dict)
    """
    stats = {}

    # Load images
    start = time.time()
    images = load_images(image_files, size=512, force_1024=True)
    imgs_ori = [(img['img_ori'].squeeze(0).permute(1,2,0)+1.)/2. for img in images]
    stats['load_images'] = time.time() - start

    # DUSt3R reconstruction
    start = time.time()
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, dust3r_model, device, batch_size=1)
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    scene.compute_global_alignment(init='mst', niter=300, schedule='linear', lr=0.01)
    scene = scene.clean_pointcloud()
    stats['dust3r_reconstruction'] = time.time() - start

    # Get camera poses and point clouds
    start = time.time()
    c2ws = scene.get_im_poses().detach()
    focals = scene.get_focals().detach()
    pps = scene.get_principal_points().detach()
    pcd = [p.detach() for p in scene.get_pts3d(clip_thred=1.0)]
    imgs = np.array(scene.imgs)

    # Masks for cleaner point cloud
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(3.0)))
    masks = scene.get_masks()
    depth = scene.get_depthmaps()
    bgs_mask = [dpt > 0.8*(torch.max(dpt[40:-40,:])+torch.min(dpt[40:-40,:])) for dpt in depth]
    masks = to_numpy([m+mb for m, mb in zip(masks, bgs_mask)])

    H, W = int(images[0]['true_shape'][0][0]), int(images[0]['true_shape'][0][1])
    stats['extract_geometry'] = time.time() - start

    # Interpolate camera poses and render
    start = time.time()
    interp_c2ws, interp_focals, interp_pps = interpolate_cameras(
        c2ws, focals, pps, num_frames, device
    )

    renderer = setup_renderer(interp_c2ws, interp_focals, interp_pps, H, W, device)
    render_results = render_pcd(pcd, imgs, masks, renderer, device)

    # Resize to 576x1024
    render_results = F.interpolate(
        render_results.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False
    ).permute(0,2,3,1)

    # Replace keyframes with originals
    keyframe_indices = np.linspace(0, num_frames-1, len(imgs_ori), dtype=int)
    for idx, img_ori in zip(keyframe_indices, imgs_ori):
        render_results[idx] = img_ori
    stats['render_pointcloud'] = time.time() - start

    # Run diffusion refinement
    start = time.time()
    diffusion_results = run_diffusion_chunks(render_results, diffusion_model, num_frames, ddim_steps, device)
    stats['diffusion_refinement'] = time.time() - start

    if return_stats:
        return diffusion_results, stats
    else:
        return diffusion_results
