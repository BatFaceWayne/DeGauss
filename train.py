#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
"""
Dynamic Gaussian Splatting training script (release version).
This refactor only adds concise documentation and removes duplicate imports
while keeping the original variable names and overall structure intact.
"""
import torch.nn as nn
from PIL import Image

import cv2
import numpy as np
import random
import os, sys
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, entropy_loss, structural_ssim, \
    ssim_raw, EdgeAwareTV

from gaussian_renderer import render_background, render_foreground, render_mask
# import sys  # Duplicate import removed
from scene import Scene, GaussianModel, GaussianModel_dynamic, Scene2gs_mixed
from utils.general_utils import safe_state
import uuid
import torch.nn.functional as F
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer
from utils.loader_utils import FineSampler, get_stamp_list
import lpips
from utils.scene_utils import render_training_image
from time import time
import copy
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt  # Duplicate import removed
from matplotlib import font_manager

font_manager.fontManager.addfont('./utils/Times New Roman.ttf')

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def get_edge_mask(image, threshold=0.1, dilation_iterations=2):
    """Return a binary edge map using Sobel magnitude.

    Parameters
    ----------
    image : torch.Tensor (B,C,H,W)
        Input RGB image in range [0,1].
    threshold : float, optional
        Normalised gradient magnitude above which a pixel is considered edge.
    dilation_iterations : int, optional
        Number of binary dilation steps to slightly expand the edges.  Currently
        disabled to avoid extra dependency cost.
    """

    from scipy.ndimage import binary_dilation

    # --- Sobel kernels -----------------------------------------------------
    sobel_x = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).unsqueeze(0).unsqueeze(0).to(image.device)
    sobel_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).unsqueeze(0).unsqueeze(0).to(image.device)

    edge_masks = []
    for c in range(image.size(1)):  # Apply Sobel filter to each channel separately
        channel = image[:, c:c + 1, :, :]
        grad_x = F.conv2d(channel, sobel_x, padding=1)
        grad_y = F.conv2d(channel, sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        edge_masks.append(grad_mag)

    # Combine edge masks from all channels
    edge_mask = torch.max(torch.stack(edge_masks, dim=1), dim=1)[0]

    # Create binary edge mask (thresholding the gradient magnitude)
    edge_mask = (edge_mask > threshold).float()

    # Dilate the edge mask
    edge_mask_np = edge_mask.cpu().numpy()
    # for i in range(edge_mask_np.shape[0]):
    #     edge_mask_np[i] = binary_dilation(edge_mask_np[i], iterations=dilation_iterations)

    return torch.from_numpy(edge_mask_np).to(image.device).float()

def normalize_depth(depth_i):
    """Normalise a depth map to [0,1] for visualisation.

    This helper avoids division-by-zero by adding an epsilon to the denominator.
    """
    depth_max = depth_i.max()
    depth_min = depth_i.min()
    return ((depth_i - depth_min) / (depth_max - depth_min + 1e-9))


class BrightnessActivation(nn.Module):
    """Piece-wise linear brightness mapping used by the light-control branch."""
    def __init__(self):
        super(BrightnessActivation, self).__init__()

    def forward(self, x):
        """Map raw [0,1] probability → perceptually-balanced brightness factor.

        Below a 0.75 threshold the mapping is identity (no amplification).  Above
        the threshold we linearly re-scale to the range [0.75,10] to give the
        optimisation headroom when compensating for very dark backgrounds.
        """
        output = torch.zeros_like(x)
        mask1 = (x <= 0.75)
        mask2 = (x > 0.75)

        # Linear part for x in [0, 0.75]
        output[mask1] = x[mask1]

        # Linear transformation for x in (0.75, 1]
        output[mask2] = 0.75 + (x[mask2] - 0.75) * ((10 - 0.75) / (1 - 0.75))

        return output

def scene_reconstruction_degauss(dataset, optimization_params, hypernetwork_config, pipeline_config, testing_iterations, saving_iterations,
                                       checkpoint_iterations, checkpoint, debug_from,
                                       foreground_gaussians, scene, stage, tb_writer, train_iter, timer, background_gaussians=None,
                                       expname='debug_2gs'):
    # ---------------------------------------------------------------------
    #  ➤  COARSE / FINE TRAINING LOOP (Foreground & Background Gaussians)
    # ---------------------------------------------------------------------
    """Two-stage (coarse→fine) training loop for DeGauss dynamic Gaussian splatting.

    This routine orchestrates optimisation of *foreground* (dynamic) and
    *background* (static) Gaussian models by alternating between:

    1.  Coarse stage – establishes scene geometry, aggressive densification.
    2.  Fine stage  – refines colour/opacity, enforces temporal & depth priors.

    It supports mixed-resolution datasets, gradient accumulation, adaptive
    densification/pruning, brightness control, and a rich loss cocktail.

    Parameters
    ----------
    dataset : ModelParams-ready dataset object
    opt : argparse.Namespace
        All optimisation hyper-parameters.
    hyper : ModelHiddenParams
        Deformation & temporal smoothness coefficients.
    pipe : PipelineParams
        Rendering pipeline configuration.
    testing_iterations / saving_iterations / checkpoint_iterations : list[int]
        Iteration indices for eval / lightweight save / full checkpoint.
    checkpoint : str | None
        Path to an existing checkpoint to resume from.
    debug_from : int
        Earliest iteration at which extra debug visualisations are produced.
    gaussians : GaussianModel_dynamic
        Foreground (dynamic) Gaussian representation.
    scene : Scene2gs_mixed
        Wrapper that holds cameras + both Gaussian sets.
    stage : {"coarse", "fine"}
    tb_writer : SummaryWriter | None
    train_iter : int
        Number of optimisation iterations for this stage.
    timer : utils.Timer
    gaussians_second : GaussianModel, optional
        Background (static) Gaussian representation.
    expname : str
        Experiment tag used for output folders.
    """
    first_iter = 0

    # ---- Model Setup ----
    foreground_gaussians.training_setup(optimization_params)
    background_gaussians.training_setup(optimization_params)

    # ---- Checkpoint Resume Logic ----
    if checkpoint:
        if stage == "coarse" and stage not in checkpoint:
            print("start from fine stage, skip coarse stage.")
            return
        if stage in checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            foreground_gaussians.restore(model_params, optimization_params)

    # ---- Background Colors (near-black to avoid numerical issues) ----
    bg_color = [1e-7, 1e-7, 1e-7]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    black_color = [1e-7, 1e-7, 1e-7]
    black_bg = torch.tensor(black_color, dtype=torch.float32, device="cuda")

    # ---- CUDA Timing Events ----
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    # ---- Gradient Accumulation Setup ----
    if stage == "coarse":
        accumulation_steps = 1  # No accumulation in coarse stage
    else:
        accumulation_steps = optimization_params.accumulation_steps

    # ---- Training State Initialization ----
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    ema_lossl1_for_log = 0.0

    final_iter = train_iter
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1

    # ---- Camera Data Loading ----
    video_cams = scene.getVideoCameras()
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()
    val_cams = scene.getValCameras()

    viewpoint_stack_index = list(range(len(train_cams)))
    if not viewpoint_stack and not optimization_params.dataloader:
        # Manual sampling mode - copy camera list
        viewpoint_stack = [i for i in train_cams]
        viewpoint_stack_index_save = copy.deepcopy(viewpoint_stack_index)

    batch_size = optimization_params.batch_size
    print("data loading done")

    # ---- DataLoader Setup (optional) ----
    if optimization_params.dataloader:
        viewpoint_stack = scene.getTrainCameras()
        if optimization_params.custom_sampler is not None:
            sampler = FineSampler(viewpoint_stack)
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size, sampler=sampler, num_workers=16,
                                                collate_fn=list)
            random_loader = False
        else:
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size, shuffle=True, num_workers=16,
                                                collate_fn=list)
            random_loader = True
        loader = iter(viewpoint_stack_loader)



    # ---- Image Masking Setup ----
    actual_height = train_cams[0].image_height
    actual_width = train_cams[0].image_width

    # Camera-specific pixel mask (invalid regions)
    if optimization_params.camera_mask:
        pixel_valid_mask = cv2.imread(
            optimization_params.camera_mask,
            cv2.IMREAD_GRAYSCALE)

        pixel_valid_mask = cv2.resize(pixel_valid_mask, (actual_width, actual_height), interpolation=cv2.INTER_NEAREST)
        pixel_valid_mask = torch.from_numpy(pixel_valid_mask).unsqueeze(0).unsqueeze(0).float().cuda() > 0
    else:
        pixel_valid_mask = torch.ones(actual_height, actual_width ).unsqueeze(0).unsqueeze(0).float().cuda()

    # Vignette correction mask (lens distortion)
    if optimization_params.vignette_mask:

        vignette_rgb_path = optimization_params.vignette_mask
        vignette_rgb = Image.open(vignette_rgb_path)
        offset_x, offset_y = (32, 32)
        w, h, = vignette_rgb.size
        vignette_img = vignette_rgb.crop((offset_x, offset_y, w - offset_x, h - offset_y))
        vignette_img = vignette_img.resize((actual_width, actual_height))
        vignette_img_tensor = torch.from_numpy(np.array(vignette_img)[:, :, :3]).float().cuda() / 255

        vignette_mask = vignette_img_tensor.permute(2, 0, 1).unsqueeze(0)

    # ---- Special Coarse Stage Initialization ----
    if stage == "coarse" and optimization_params.zerostamp_init:
        # Filter cameras to timestamp 0 only
        temp_list = get_stamp_list(viewpoint_stack, 0)
        viewpoint_stack = temp_list.copy()
    else:
        load_in_memory = False

    # ---- Main Training Loop ----
    Finish_whose_seq = False
    for iteration in range(first_iter, final_iter + 1):
        iter_start.record()

        # Learning rate scheduling
        foreground_gaussians.update_learning_rate(iteration)
        background_gaussians.update_learning_rate(iteration)

        # Spherical harmonics degree progression
        if iteration % optimization_params.foreground_oneupshinterval == 0:
            foreground_gaussians.oneupSHdegree()
        if iteration % optimization_params.background_oneupshinterval == 0:
            background_gaussians.oneupSHdegree()

        # ---- Camera Batch Sampling ----
        if optimization_params.dataloader:
            try:
                viewpoint_cams = next(loader)
            except StopIteration:
                print("reset dataloader into random dataloader.")
                if not random_loader:
                    viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=optimization_params.batch_size, shuffle=True,
                                                        num_workers=32, collate_fn=list)
                    random_loader = True
                loader = iter(viewpoint_stack_loader)

        else:
            idx = 0
            viewpoint_cams = []
            # Mixed-resolution handling (currently disabled)
            handle_different_cam = False

            if not handle_different_cam:
                # Simple random sampling without replacement
                while idx < batch_size:
                    viewpoint_cam_idx = viewpoint_stack_index.pop(randint(0, len(viewpoint_stack_index) - 1))
                    viewpoint_cam = viewpoint_stack[viewpoint_cam_idx]
                    if not viewpoint_stack_index:
                        # Reset for next epoch
                        viewpoint_stack_index = viewpoint_stack_index_save.copy()
                        Finish_whose_seq = True
                    viewpoint_cams.append(viewpoint_cam)
                    idx += 1
            else:
                viewpoint_cams = []
                current_resolution = None  # Track the resolution for the current batch
                batch_resolved = False

                while idx < batch_size:

                    viewpoint_cam_idx = viewpoint_stack_index.pop(randint(0, len(viewpoint_stack_index) - 1))

                    viewpoint_cam = viewpoint_stack[viewpoint_cam_idx]

                    # Get the resolution of the current viewpoint camera
                    cam_resolution = viewpoint_cam.image_width  # Assuming viewpoint_cam has a method to get its resolution

                    if current_resolution is None:
                        # Set the resolution for the batch if not yet set
                        current_resolution = cam_resolution

                    # Check if the viewpoint_cam has the same resolution as the current batch resolution
                    if cam_resolution == current_resolution:
                        viewpoint_cams.append(viewpoint_cam)
                        idx += 1
                    else:
                        # If a different resolution is encountered, skip this camera
                        viewpoint_stack_index.append(viewpoint_cam_idx)

                    # Check if we are running out of images with the current resolution
                    remaining_same_resolution = sum(1 for cam_idx in viewpoint_stack_index if
                                                    viewpoint_stack[cam_idx].image_width == current_resolution)

                    if remaining_same_resolution + len(viewpoint_cams) < batch_size:
                        # If there aren't enough same-resolution images left to fill the batch, exit the loop
                        print(f"Exiting: Not enough images with resolution {current_resolution} to fill the batch.")
                        idx += 1

                    if not viewpoint_stack_index:
                        viewpoint_stack_index = viewpoint_stack_index_save.copy()
                        Finish_whose_seq = True

                # Continue the rest of your logic

            if len(viewpoint_cams) == 0:
                continue

        # ---- Batch Rendering Setup ----
        images = []
        gt_images = []
        images_second = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        viewspace_point_tensor_list_motion = []
        radii_list_second = []
        visibility_filter_list_second = []
        viewspace_point_tensor_list_second = []
        motion_masks = []
        foreground_prob_list = []
        depth_images = []
        depth_images_dy = []

        # ---- Multi-Model Rendering Loop ----
        for viewpoint_cam in viewpoint_cams:

            # Render foreground (dynamic) Gaussians
            render_pkg_dynamic_pers = render_foreground(viewpoint_cam, foreground_gaussians, pipeline_config, black_bg, stage=stage,
                                                     cam_type=scene.dataset_type)

            image, viewspace_point_tensor, visibility_filter, radii = render_pkg_dynamic_pers["render"], \
                render_pkg_dynamic_pers[
                    "viewspace_points"], render_pkg_dynamic_pers["visibility_filter"], render_pkg_dynamic_pers["radii"]
            images.append(image.unsqueeze(0))
            depth_images_dy.append(render_pkg_dynamic_pers['depth'].unsqueeze(0))

            # Ground truth image
            gt_image = viewpoint_cam.original_image.float().cuda() / 255

            # Render background (static) Gaussians
            render_pkg_second = render_background(viewpoint_cam, background_gaussians, pipeline_config, black_bg, stage='coarse',
                                       cam_type=scene.dataset_type)

            # Render motion probability mask
            render_pkg_motion = render_mask(viewpoint_cam, foreground_gaussians, pipeline_config, background, stage=stage,
                                              cam_type=scene.dataset_type)
            image_second = render_pkg_second["render"]
            motion_mask = render_pkg_motion["render"]
            motion_masks.append(motion_mask.unsqueeze(0))
            images_second.append(image_second.unsqueeze(0))
            depth_images.append(render_pkg_second['depth'].unsqueeze(0))
            foreground_prob_list.append(render_pkg_motion["foreground_prob"].unsqueeze(0))

            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
            viewspace_point_tensor_list_motion.append(render_pkg_motion['viewspace_points'])
            radii_list_second.append(render_pkg_second['radii'].unsqueeze(0))
            viewspace_point_tensor_list_second.append(render_pkg_second['viewspace_points'])
            visibility_filter_list_second.append(render_pkg_second['visibility_filter'].unsqueeze(0))

        # ---- Batch Tensor Consolidation ----
        radii = torch.cat(radii_list, 0).max(dim=0).values
        motion_masks = torch.cat(motion_masks, 0)
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)

        # Accumulate visibility filters for background model
        if (iteration - 1) % accumulation_steps == 0:
            visibility_filter_second = torch.cat(visibility_filter_list_second).any(dim=0)
        else:
            visibility_filter_second = torch.logical_or(visibility_filter_second,torch.cat(visibility_filter_list_second).any(dim=0))

        image_tensor_first = torch.cat(images, 0)
        gt_image_tensor = torch.cat(gt_images, 0)
        image_tensor_second = torch.cat(images_second, 0)
        radii_second = torch.cat(radii_list_second, 0).max(dim=0).values
        depth_images_tensor = torch.cat(depth_images, 0)
        depth_images_dy_tensor = torch.cat(depth_images_dy, 0)
        foreground_prob_tensor = torch.cat(foreground_prob_list, 0).max(dim=0).values.cuda()

        # ---- Pruning Schedule Configuration ----
        if optimization_params.prune_small_foreground_visbility:
            step_to_prune = optimization_params.iterations / 2
        else:
            # Never prune based on visibility (2x max iterations)
            step_to_prune = optimization_params.iterations * 2

        # ---- Motion Probability Processing ----
        if stage == "coarse":
            motion_pro_first = motion_masks[:, 2:3, :, :]
            motion_pro_second = motion_masks[:, 1:2, :, :]
        else:
            motion_pro_first = motion_masks[:, 2:3, :, :] + 1e-6
            motion_pro_second = motion_masks[:, 1:2, :, :] + 1e-6

        # ---- Apply Image Masks ----
        if optimization_params.vignette_mask:
            image_tensor_second = image_tensor_second * pixel_valid_mask * vignette_mask
            gt_image_tensor = gt_image_tensor * pixel_valid_mask
            image_tensor_first = image_tensor_first * pixel_valid_mask * vignette_mask
        else:
            image_tensor_second = image_tensor_second * pixel_valid_mask
            gt_image_tensor = gt_image_tensor * pixel_valid_mask
            image_tensor_first = image_tensor_first * pixel_valid_mask

        # ---- Brightness Control Activation ----
        activation_light = BrightnessActivation()
        light_var = 0.5 + activation_light(motion_masks[:, 0:1, :, :].repeat(1, 3, 1, 1))

        # ---- Background Brightness Adjustment ----
        if optimization_params.use_brightness_control:
            # Save raw background for debugging/loss computation
            image_second_to_show = image_tensor_second.clone().detach().cpu()
            image_second_to_show_g = image_tensor_second.clone().detach()
            image_tensor_second = image_tensor_second * light_var

            image_tensor_second = torch.clamp(image_tensor_second, 0, 1 - 1e-9)
            image_tensor_first = image_tensor_first
            image_tensor_first = torch.clamp(image_tensor_first, 0, 1 - 1e-9)

        else:
            image_second_to_show = image_tensor_second.clone().detach().cpu()
            image_second_to_show_g = image_tensor_second.clone().detach()

        # ---- Probabilistic Motion Mask Normalization ----
        distri = motion_pro_first + motion_pro_second
        motion_masks_first = motion_pro_first / distri
        motion_masks_second = motion_pro_second / distri

        motion_masks_first = motion_masks_first * pixel_valid_mask
        motion_masks_second = motion_masks_second * pixel_valid_mask

        # Motion mask aliases for loss computation
        mask_comp_first = motion_masks_first
        mask_comp_second = motion_masks_second

        # ---- Image Composition ----
        image_dy = image_tensor_first * motion_masks_first
        image_sta = image_tensor_second * motion_masks_second

        vis_thresh = 0.49

        # Final composite image
        image_tensor = image_dy + image_sta

        # ---- Primary Loss Computation ----
        Ll1 = l1_loss(image_tensor, gt_image_tensor[:, :3, :, :])
        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()

        # ---- Stage-Specific Loss ----
        if stage == "coarse":
            # Coarse stage: background + mixed component loss

            ll2 = l1_loss(image_tensor_second, gt_image_tensor[:, :3, :, :])
            ###### override with 0.5 compoisition
            ##### LL1 learns 0.9 ground truth to regularize foreground learning but allowing structrual modeling
            if optimization_params.coarse_mean_override:
                Ll1 = l1_loss(
                    image_tensor_first * (0.5) + image_tensor_second.clone().detach() * (
                        0.5)
                    , gt_image_tensor[:, :3, :, :] * 0.9)
            else:
                Ll1 = l1_loss(
                    image_tensor_first * (motion_masks_first) + image_tensor_second.clone().detach() * (motion_masks_second)
                    , gt_image_tensor[:, :3, :, :] * 0.9)
            loss = optimization_params.lambda_main_loss * Ll1 + optimization_params.lambda_main_loss * ll2


        else:
            # Fine stage: focus on composite quality
            loss = optimization_params.lambda_main_loss * Ll1
 
        # ---- Brightness Regularization Loss ----
        start_to_penal = optimization_params.densify_until_iter // 10
        loss_light = 0.0001 * l1_loss(light_var * pixel_valid_mask, torch.ones_like(light_var).cuda() * pixel_valid_mask)

        weight_penal_light_start = 0.001
        weight_penal_light_end = optimization_params.weight_penal_light_end
        weight_penal_light = weight_penal_light_start + (weight_penal_light_end - weight_penal_light_start) * (
                iteration - start_to_penal) / (optimization_params.densify_until_iter - start_to_penal)
        if optimization_params.explicitly_update_brightness_control and optimization_params.use_brightness_control:
            image_second_without_light_var = image_second_to_show_g.clone().detach()
            this_light_var = torch.clamp(image_second_without_light_var * light_var, 0, 1 - 1e-9)
            if iteration<=start_to_penal:
                loss_light += 0.01 * l1_loss(this_light_var, gt_image_tensor)
            else:
                if weight_penal_light<0.01:
                    loss_light += (0.01-weight_penal_light) * l1_loss(this_light_var, gt_image_tensor)

        ssim_notdense = 0.4 * (1.0 - ssim(image_tensor, gt_image_tensor[:, :3, :, :]))
        loss_light += ssim_notdense

        # ---- Additional Brightness Regularization ----
        regularize_light_var = True
        if regularize_light_var and stage == "fine" and optimization_params.use_brightness_control:

            if iteration > start_to_penal:
                loss_light += weight_penal_light * l1_loss(light_var * pixel_valid_mask,
                                                     torch.ones_like(light_var).cuda() * pixel_valid_mask)



        # ---- Stage-Specific SSIM Loss ----
        if stage == 'coarse':
            ssim_loss_temp = ssim(image_tensor_second, gt_image_tensor)
            ssim_loss_temp1 = ssim(image_tensor_first, gt_image_tensor)
            loss += 0.2 * (1.0 - ssim_loss_temp)
            loss += 0.2 * (1.0 - ssim_loss_temp1)

        # ---- Temporal Smoothness Loss ----
        if stage == "fine" and hypernetwork_config.time_smoothness_weight != 0:
            tv_loss = foreground_gaussians.compute_regulation(hypernetwork_config.time_smoothness_weight, hypernetwork_config.l1_time_planes,
                                                   hypernetwork_config.plane_tv_weight)
            loss += tv_loss

        # ---- Depth Separation Losses ----
        if optimization_params.separation_high_prob:
            loss_depth_back = torch.maximum(
                (
                            depth_images_dy_tensor - depth_images_tensor.clone().detach()) / scene.cameras_extent * pixel_valid_mask * (
                            mask_comp_first > 0.6).clone().detach(),
                torch.tensor(0)).mean()
            loss += optimization_params.lambda_loss_depth_back * loss_depth_back
        if optimization_params.separation_low_prob:
            loss_depth_forward = torch.maximum(
                (
                            depth_images_tensor.clone().detach() - depth_images_dy_tensor) / scene.cameras_extent * pixel_valid_mask * (
                            mask_comp_first < 0.4).clone().detach(),
                torch.tensor(0)).mean()
            loss += 0.1 * loss_depth_forward
            if iteration % 100 == 0:
                print("depth loss", loss_depth_back.item())
                print("depth forward loss", loss_depth_forward.item())

        # ---- Depth Smoothness Loss ----
        if optimization_params.use_depth_smoothness_loss:
            depth_reg = EdgeAwareTV()
            depth_scaled = depth_images_tensor / scene.cameras_extent
            loss_smooth_sta = depth_reg(depth_scaled.permute(0, 2, 3, 1), gt_image_tensor.permute(0, 2, 3, 1))
            loss += optimization_params.lambda_depth_smoothness * (loss_smooth_sta)


        # ---- Dynamic Content Diversity Loss ----
        if optimization_params.penalize_dynamic:
            dynamic_segment = image_tensor_first
            static_disposed = (image_tensor_second ).clone().detach().cuda()
            d_ssim_str = (structural_ssim(dynamic_segment, static_disposed)) * pixel_valid_mask * (
                    mask_comp_first > 1 - vis_thresh)


            diversity_penalty = d_ssim_str.mean()
            loss += 0.01 * diversity_penalty


        # ---- Foreground/Background Component Losses ----
        if optimization_params.use_foreground_background_loss:

            # High-confidence region losses
            masked_2d_first = mask_comp_first > 1 - vis_thresh

            l1_comp_first = l1_loss(image_tensor_first * masked_2d_first, gt_image_tensor * masked_2d_first)

            if optimization_params.ssim_loss_use:
                ssim_loss_raw_first = ssim_raw(image_tensor_first, gt_image_tensor)
                loss += 0.4 * ((1.0 - (ssim_loss_raw_first)) * masked_2d_first * pixel_valid_mask).mean()

            loss += 2 * l1_comp_first
            masked_2d = mask_comp_second > 1 - vis_thresh

            l1_comp_second = l1_loss(image_tensor_second * masked_2d, gt_image_tensor * masked_2d)

            if optimization_params.ssim_loss_use:
                ssim_raw_second = ssim_raw(image_tensor_second, gt_image_tensor)
                loss += 0.4 * ((1.0 - ssim_raw_second) * masked_2d * pixel_valid_mask).mean()

            loss += 2 * l1_comp_second


        # ---- Gaussian Scale Regularization ----
        if optimization_params.use_penal_large_gaussians:
            max_scale = 0.1 * scene.cameras_extent
            scale_exp = torch.exp(foreground_gaussians._scaling)
            scale_max_size = (torch.maximum( scale_exp.amax(dim=-1),
                                            torch.tensor(max_scale).cuda()) - max_scale)
            scale_reg_max = 0.01 * scale_max_size.mean()
            loss_light += scale_reg_max
            scale_exp_second = torch.exp(background_gaussians._scaling)
            scale_max_size_second = (torch.maximum( scale_exp_second.amax(dim=-1),
                                            torch.tensor(max_scale).cuda()) - max_scale)
            scale_reg_second_max = 0.01 * scale_max_size_second.mean()
            loss_light += scale_reg_second_max

        # ---- Gaussian Aspect Ratio Regularization ----
        if optimization_params.use_penal_spiky_gaussians:
            if iteration % 10 == 0:
                scale_exp = torch.exp(foreground_gaussians._scaling)
                scale_reg = (
                        torch.maximum(
                            scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                            torch.tensor(5),
                        )
                        - 5
                )
                scale_reg = 0.1 * scale_reg.mean()
                loss_light += scale_reg
            if iteration % 10 == 0:
                scale_exp = torch.exp(background_gaussians._scaling)
                scale_reg = (
                        torch.maximum(
                            scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                            torch.tensor(5),
                        )
                        - 5
                )
                scale_reg = 0.1 * scale_reg.mean()
                loss_light += scale_reg

        # ---- Motion Mask Entropy Loss (second half of training) ----
        if iteration > optimization_params.iterations // 2:
            loss_light += optimization_params.lambda_entropy_loss * entropy_loss(mask_comp_first)
        # ---- Gradient Accumulation & Backpropagation ----
        loss = loss / (accumulation_steps)
        loss.backward(retain_graph=True)

        # Emergency restart on NaN loss
        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)

        # ---- Gradient Collection from Multi-Model Rendering ----
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        if (iteration-1) % accumulation_steps == 0:
            viewspace_point_tensor_grad_second = torch.zeros_like(render_pkg_second['viewspace_points'])

        for idx in range(0, len(viewspace_point_tensor_list)):
            if optimization_params.use_motion_grad:
                # Include motion mask gradients for foreground
                viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[
                    idx].grad.clone() + viewspace_point_tensor_list_motion[idx].grad.clone()
                viewspace_point_tensor_grad_second = viewspace_point_tensor_grad_second + \
                                                     viewspace_point_tensor_list_second[idx].grad.clone()
            else:
                viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad.clone()
                viewspace_point_tensor_grad_second = viewspace_point_tensor_grad_second + \
                                                     viewspace_point_tensor_list_second[idx].grad

        # ---- Brightness Loss Backpropagation ----
        if loss_light:
            loss_light = loss_light / (accumulation_steps)
            loss_light.backward()

        if iteration % 100 == 0:
            print('redner_grad: ', torch.norm((viewspace_point_tensor_list[idx].grad).clone().detach()).mean())
            print('redner_grad_motion: ',
                  torch.norm((viewspace_point_tensor_list_motion[idx].grad).clone().detach()).mean())
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            ema_lossl1_for_log = 0.4 * Ll1.item() + 0.6 * ema_lossl1_for_log
            total_point = foreground_gaussians._xyz.shape[0]
            total_point_second = background_gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "ll1": f"{ema_lossl1_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point": f"{total_point}",
                                          "point_second": f"{total_point_second}"})
                progress_bar.update(10)
            if iteration == optimization_params.iterations:
                progress_bar.close()

            # ---- Logging and Saving ----
            timer.pause()
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)

            # ---- Debug Visualization (every 100 iterations) ----
            if iteration % 100 == 0:

                out_debug_depth_dir = os.path.join(optimization_params.saving_folder, expname)
                os.makedirs(out_debug_depth_dir, exist_ok=True)

                fig, ax = plt.subplots(3, 5, figsize=(30, 18))
                plt.rcParams['font.family'] = "sans-serif"

                ax[0, 0].imshow(gt_image_tensor.clone().detach().cpu()[0].permute(1, 2, 0).numpy())
                ax[0, 0].set_title("Ground Truth")
                ax[1, 0].imshow(image_tensor.clone().detach().cpu()[0].permute(1, 2, 0).numpy())
                ax[1, 0].set_title("Full Predicted")
                # render_pkg_temp = render_egopers(viewpoint_cam, foreground_gaussians, pipeline_config, background, stage=stage,
                #                        cam_type=scene.dataset_type)
                ax[0, 1].imshow(image_tensor_first[0].clone().detach().cpu().permute(1, 2, 0).numpy())
                ax[0, 1].set_title("Dynamic Raw")

                ax[1, 1].imshow(image_dy[0].clone().detach().cpu().permute(1, 2, 0).numpy())
                ax[1, 1].set_title("Dynamic Final")

                ax[0, 2].imshow(image_second_to_show[0].permute(1, 2, 0).numpy())
                ax[0, 2].set_title("Static Raw")
                ax[1, 2].imshow(image_sta[0].clone().detach().cpu().permute(1, 2, 0).numpy())
                ax[1, 2].set_title("Static Final")
                pos = ax[0, 3].imshow(
                    (motion_masks[0, 2:3, :, :] * pixel_valid_mask[0]).clone().detach().cpu().permute(1, 2, 0).numpy(),
                    cmap='jet', vmin=0, vmax=1)
                ax[0, 3].set_title("Dynamic Prob")

                ax[1, 3].imshow(((motion_masks[0, 0:1, :,
                                  :] * pixel_valid_mask[0]).clone().detach().cpu().permute(1, 2, 0).numpy()),
                                cmap='jet',
                                vmin=0, vmax=1)
                ax[1, 3].set_title("Brightness Control")
                ax[0, 4].imshow(image_tensor_second[0].clone().detach().cpu().permute(1, 2, 0).numpy())
                ax[0, 4].set_title("Static Raw with Brightness Control")
                ax[1, 4].imshow(
                    ((torch.abs(gt_image_tensor - image_tensor_second).sum(axis=1))[0].clone().detach().cpu().numpy()),
                    cmap='jet',
                    vmin=0, vmax=1)
                ax[1, 4].set_title("Static Raw - GT Error")
                ax[2, 0].imshow(normalize_depth((depth_images_dy_tensor * mask_comp_first[0,
                                                                     :1] + depth_images_tensor * mask_comp_second[0,
                                                                                                 :1])[
                                               0].clone().detach().cpu().permute(1, 2, 0).numpy()), cmap='jet',
                                vmin=0, vmax=1)
                ax[2, 0].set_title("Depth Image Pred")
                ax[2, 1].imshow(normalize_depth((depth_images_dy_tensor)[0].clone().detach().cpu().permute(1, 2, 0).numpy()),
                                cmap='jet', vmin=0, vmax=1)
                ax[2, 1].set_title("Depth Image Pred dynamic")
                ax[2, 2].imshow(normalize_depth((depth_images_tensor)[0].clone().detach().cpu().permute(1, 2, 0).numpy()),
                                cmap='jet',
                                vmin=0, vmax=1)
                ax[2, 2].set_title("Depth Image Pred static")
                ax[2, 3].imshow(((motion_masks[0, 1:2, :,
                                  :] * pixel_valid_mask[0]).clone().detach().cpu().permute(1, 2, 0).numpy()),
                                cmap='jet',
                                vmin=0, vmax=1)
                ax[2, 3].set_title("Static Raw Prob")
                ax[2, 4].imshow(
                    ((mask_comp_first[0] * pixel_valid_mask[0])).clone().detach().cpu().permute(1, 2, 0).numpy(),
                    cmap='jet', vmin=0, vmax=1)
                ax[2, 4].set_title("Full Probabilistic Mask")

                plt.savefig(os.path.join(out_debug_depth_dir, stage + '_' + str(iteration).zfill(6) + ".jpg"))
                plt.close()


            timer.start()
            # ---- Adaptive Gaussian Densification & Pruning ----
            if iteration < optimization_params.densify_until_iter:

                # Track foreground importance (using probability instead of radii)
                foreground_gaussians.max_radii2D[visibility_filter] = torch.max(foreground_gaussians.max_radii2D[visibility_filter],
                                                                      foreground_prob_tensor[visibility_filter])
                foreground_gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                # ---- Threshold Scheduling ----
                if stage == "coarse":
                    opacity_threshold = optimization_params.opacity_threshold_coarse
                    densify_threshold = optimization_params.densify_grad_threshold_coarse
                else:
                    # Progressive threshold scheduling for fine stage
                    opacity_threshold = optimization_params.opacity_threshold_fine_init - iteration * (
                            optimization_params.opacity_threshold_fine_init - optimization_params.opacity_threshold_fine_after) / (
                                            optimization_params.densify_until_iter)
                    if iteration < optimization_params.densify_until_iter * 0.5:
                        # First half: gradually reduce threshold
                        densify_threshold = optimization_params.densify_grad_threshold_fine_init - iteration * (
                                optimization_params.densify_grad_threshold_fine_init - optimization_params.densify_grad_threshold_after) / (
                                                optimization_params.densify_until_iter)
                    else:
                        # Second half: use minimum threshold
                        densify_threshold = optimization_params.densify_grad_threshold_after
                    if accumulation_steps>1:
                        # Scale threshold for gradient accumulation
                        densify_threshold = densify_threshold / (accumulation_steps )
                    if optimization_params.make_foreground_thresh_larger:
                        # Make foreground densification more aggressive
                        densify_threshold = densify_threshold * 2

                # ---- Densification (Add Gaussians) ----
                if iteration > optimization_params.densify_from_iter and iteration % optimization_params.densification_interval == 0 and \
                        foreground_gaussians.get_xyz.shape[0] < optimization_params.max_gaussian_foreground:
                    size_threshold = 50 if iteration > optimization_params.opacity_reset_interval else None

                    foreground_gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold, 5, 5,
                                      scene.model_path, iteration, stage)

                # ---- Pruning (Remove Gaussians) ----
                if iteration > optimization_params.pruning_from_iter and iteration % optimization_params.pruning_interval == 0 and \
                        foreground_gaussians.get_xyz.shape[0] > 10000:
                    size_threshold = 20 if iteration > step_to_prune else None
                    if Finish_whose_seq:
                        # End-of-epoch pruning with probability-based criteria
                        run_pruning_prob = True
                        foreground_gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold,
                                        run_pruning_prob=run_pruning_prob)
                        Finish_whose_seq = False
                    else:
                        # Standard pruning
                        foreground_gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)

                # ---- Optional Point Growing ----
                if iteration % optimization_params.densification_interval == 0 and foreground_gaussians.get_xyz.shape[
                    0] < 360000 and optimization_params.add_point:
                    foreground_gaussians.grow(5, 5, scene.model_path, iteration, stage)

                # ---- Opacity Reset (prevent degradation) ----
                if iteration % optimization_params.opacity_reset_interval == 0:
                    print("reset opacity")
                    foreground_gaussians.reset_opacity_partially_small()
                    if optimization_params.reset_SH:
                        foreground_gaussians.reset_sh_partially_first()

            # ---- Foreground Model Optimization ----
            max_norm = 1.0
            # Optional gradient clipping (currently disabled)
            # torch.nn.utils.clip_grad_norm_(foreground_gaussians._deformation.parameters(), max_norm) 
            if iteration < optimization_params.iterations:
                foreground_gaussians.optimizer.step()
                foreground_gaussians.optimizer.zero_grad(set_to_none=True)

            # ---- Background Model Management ----
            if iteration < optimization_params.densify_until_iter and iteration % accumulation_steps == 0:
                visibility_filter = visibility_filter_second
                radii = radii_second

                # Track background importance (using actual radii)
                background_gaussians.max_radii2D[visibility_filter] = torch.max(
                    background_gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter])
                background_gaussians.add_densification_stats(viewspace_point_tensor_grad_second, visibility_filter)

                # ---- Background Threshold Configuration ----
                if stage == "coarse":
                    opacity_threshold = optimization_params.opacity_threshold_coarse
                    densify_threshold = optimization_params.densify_grad_threshold_coarse
                else:
                    opacity_threshold = optimization_params.opacity_threshold_fine_init - iteration * (
                            optimization_params.opacity_threshold_fine_init - optimization_params.opacity_threshold_fine_after) / (
                                            optimization_params.densify_until_iter)
                    # Background uses coarse threshold (less aggressive)
                    densify_threshold = optimization_params.densify_grad_threshold_coarse
                    if optimization_params.make_background_thresh_larger:
                        densify_threshold = densify_threshold * 2

                # ---- Background Densification ----
                if iteration > optimization_params.densify_from_iter and iteration % optimization_params.densification_interval == 0 and \
                        background_gaussians.get_xyz.shape[0] < optimization_params.max_gaussian_background:
                    size_threshold = 20 if iteration > optimization_params.opacity_reset_interval else None

                    background_gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent,
                                                 size_threshold,
                                                 5, 5,
                                                 scene.model_path, iteration, stage)

                # ---- Background Pruning ----
                if iteration > optimization_params.pruning_from_iter and iteration % optimization_params.pruning_interval == 0 and \
                        background_gaussians.get_xyz.shape[0] > 100000:
                    size_threshold = 50 if iteration > step_to_prune and iteration < int(
                        optimization_params.densify_until_iter * 0.8) else None

                    background_gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)

                # ---- Background Opacity Reset ----
                if iteration % optimization_params.opacity_reset_interval == 0:
                    print("reset opacity")
                    background_gaussians.reset_opacity_partially()
                    if optimization_params.reset_SH:
                        background_gaussians.reset_sh_partially_second()

            # ---- Background Model Optimization ----
            if iteration < optimization_params.iterations and iteration % accumulation_steps == 0:
                background_gaussians.optimizer.step()
                background_gaussians.optimizer.zero_grad(set_to_none=True)

            # ---- Checkpoint Saving ----
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((foreground_gaussians.capture(), iteration),
                           scene.model_path + "/chkpnt" + f"_{stage}_" + str(iteration) + ".pth")
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((background_gaussians.capture(), iteration),
                           scene.model_path + "/chkpnt" + f"_{stage}_gs2_" + str(iteration) + ".pth")

    # ---- Post-Training Evaluation (Fine Stage Only) ----
    if stage == "fine":

        batch_size = 1
        viewpoint_stack_index = list(range(len(train_cams)  + len(test_cams)))
        # if not viewpoint_stack and not opt.dataloader:
        # dnerf's branch
        del viewpoint_stack
        if optimization_params.eval_include_train_cams:
            viewpoint_stack_index = list(range(len(train_cams)  + len(test_cams)))
            data_list = [0] * len(train_cams)  + [2] * len(test_cams)
            viewpoint_stack = [i for i in train_cams] + [i for i in test_cams]
        else:
            viewpoint_stack_index = list(range(len(test_cams)))
            data_list = [2] * len(test_cams)
            viewpoint_stack = [i for i in test_cams]

        for iteration in range(0, len(viewpoint_stack)):


            iter_start.record()


            # dynerf's branch
            if optimization_params.dataloader:
                try:
                    viewpoint_cams = next(loader)
                except StopIteration:
                    print("reset dataloader into random dataloader.")
                    if not random_loader:
                        viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=optimization_params.batch_size,
                                                            shuffle=False,
                                                            num_workers=32, collate_fn=list)
                        random_loader = True
                    loader = iter(viewpoint_stack_loader)

            else:
                idx = 0
                viewpoint_cams = []

                while idx < batch_size:
                    try:
                        viewpoint_cam_idx = viewpoint_stack_index.pop(0)
                    except:
                        return 0
                    viewpoint_cam = viewpoint_stack[viewpoint_cam_idx]
                    which_type = data_list[viewpoint_cam_idx]
                    viewpoint_cams.append(viewpoint_cam)
                    idx += 1
                if len(viewpoint_cams) == 0:
                    continue

            with torch.no_grad():
                images = []
                gt_images = []
                images_second = []
                radii_list = []
                visibility_filter_list = []
                viewspace_point_tensor_list = []
                radii_list_second = []
                visibility_filter_list_second = []
                viewspace_point_tensor_list_second = []
                motion_masks = []
                for viewpoint_cam in viewpoint_cams:
                    render_pkg_dynamic_pers = render_foreground(viewpoint_cam, foreground_gaussians, pipeline_config, black_bg, stage=stage,
                                                             cam_type=scene.dataset_type)
                    image, viewspace_point_tensor, visibility_filter, radii = render_pkg_dynamic_pers["render"], \
                        render_pkg_dynamic_pers[
                            "viewspace_points"], render_pkg_dynamic_pers["visibility_filter"], render_pkg_dynamic_pers[
                        "radii"]
                    images.append(image.unsqueeze(0))

                    gt_image = viewpoint_cam.original_image.float().cuda() / 255

                    render_pkg_second = render_background(viewpoint_cam, background_gaussians, pipeline_config, black_bg, stage='coarse',
                                               cam_type=scene.dataset_type)
                    image_second = render_pkg_second["render"]

                    render_pkg_motion = render_mask(viewpoint_cam, foreground_gaussians, pipeline_config, background, stage=stage,
                                                      cam_type=scene.dataset_type)

                    motion_mask = render_pkg_motion["render"]
                    motion_masks.append(motion_mask.unsqueeze(0))
                    images_second.append(image_second.unsqueeze(0))

                    gt_images.append(gt_image.unsqueeze(0))
                    radii_list.append(radii.unsqueeze(0))
                    visibility_filter_list.append(visibility_filter.unsqueeze(0))
                    viewspace_point_tensor_list.append(viewspace_point_tensor)

                    radii_list_second.append(render_pkg_second['radii'].unsqueeze(0))
                    viewspace_point_tensor_list_second.append(render_pkg_second['viewspace_points'])
                    visibility_filter_list_second.append(render_pkg_second['visibility_filter'].unsqueeze(0))

                motion_masks = torch.cat(motion_masks, 0)
                image_tensor_first = torch.cat(images, 0)
                gt_image_tensor = torch.cat(gt_images, 0)
                image_tensor_second = torch.cat(images_second, 0)
                pixel_valid_mask = torch.ones_like(gt_image_tensor)[0, 0].unsqueeze(0).unsqueeze(0).float().cuda()
                motion_masks = torch.clamp(motion_masks, 1e-9, 1 - 1e-9)
                motion_pro_first = motion_masks[:, 2:3, :, :] + 1e-6
                motion_pro_second = motion_masks[:, 1:2, :, :] + 1e-6


                if optimization_params.vignette_mask:
                    image_tensor_second = image_tensor_second * pixel_valid_mask * vignette_mask
                    gt_image_tensor = gt_image_tensor * pixel_valid_mask
                    image_tensor_first = image_tensor_first * pixel_valid_mask * vignette_mask
                else:
                    image_tensor_second = image_tensor_second * pixel_valid_mask
                    gt_image_tensor = gt_image_tensor * pixel_valid_mask
                    image_tensor_first = image_tensor_first * pixel_valid_mask

                activation_light = BrightnessActivation()

                light_var = 0.5 + activation_light(motion_masks[:, 0:1, :, :].repeat(1, 3, 1, 1))

                white_thresh = 0.9


                if optimization_params.use_brightness_control:
                    image_second_to_show = image_tensor_second.clone().detach().cpu()
                    image_tensor_second = image_tensor_second * light_var
                    image_tensor_second = torch.clamp(image_tensor_second, 0, 1 - 1e-9)

                    image_tensor_first = image_tensor_first
                    image_tensor_first = torch.clamp(image_tensor_first, 0, 1 - 1e-9)

                else:
                    image_second_to_show = image_tensor_second.clone().detach().cpu()

                distri = motion_pro_first + motion_pro_second + 1e-8
                motion_masks_first = motion_pro_first / distri
                motion_masks_second = motion_pro_second / distri

                # motion_masks_first = motion_pro_first  + 1e-6
                # motion_masks_second =  1 - motion_pro_first + 1e-6




                mask_comp_first = motion_masks_first

                image_dy = image_tensor_first * motion_masks_first
                image_sta = image_tensor_second * motion_masks_second
                image_tensor = image_dy + image_sta

                if iteration == optimization_params.iterations:
                    progress_bar.close()

                if iteration % 1 == 0:

                    out_debug_depth_dir = os.path.join(optimization_params.saving_folder, expname,
                                                       'train_cams')

                    out_debug_gt = os.path.join(optimization_params.saving_folder, expname, 'gt')
                    os.makedirs(out_debug_gt, exist_ok=True)
                    cv2.imwrite(os.path.join(out_debug_gt, viewpoint_cam.image_name + ".png"),
                                (torch.clamp(gt_image_tensor[0].clone().detach().cpu(), 0, 1).permute(1, 2,
                                                                                                      0).numpy() * 255).astype(
                                    np.uint8)[:, :, ::-1])
                    out_debug_full_pred = os.path.join(optimization_params.saving_folder, expname,
                                                       'full_pred')
                    os.makedirs(out_debug_full_pred, exist_ok=True)
                    cv2.imwrite(
                        os.path.join(out_debug_full_pred, viewpoint_cam.image_name + ".png"),
                        (torch.clamp(image_tensor[0].clone().detach().cpu(), 0, 1).permute(1, 2,
                                                                                           0).numpy() * 255).astype(
                            np.uint8)[:, :, ::-1])
                    out_debug_mask = os.path.join(optimization_params.saving_folder, expname,
                                                  'mask_comp')
                    os.makedirs(out_debug_mask, exist_ok=True)
                    feature_out_path = os.path.join(out_debug_mask, viewpoint_cam.image_name + ".npy")
                    with open(feature_out_path, "wb") as fout:
                        np.save(fout, ((mask_comp_first[0] * pixel_valid_mask[0])).clone().detach().cpu().permute(1, 2,
                                                                                                                  0).numpy())

                    out_debug_static_raw = os.path.join(optimization_params.saving_folder, expname,
                                                        'static_raw')
                    os.makedirs(out_debug_static_raw, exist_ok=True)
                    cv2.imwrite(
                        os.path.join(out_debug_static_raw, viewpoint_cam.image_name + ".png"),
                        (torch.clamp(image_second_to_show[0].clone().detach().cpu(), 0, 1).permute(1, 2,
                                                                                                   0).numpy() * 255).astype(
                            np.uint8)[:, :, ::-1])

                    os.makedirs(out_debug_depth_dir, exist_ok=True)

                    out_debug_static_raw = os.path.join(optimization_params.saving_folder, expname,
                                                        'static_light')
                    os.makedirs(out_debug_static_raw, exist_ok=True)
                    cv2.imwrite(
                        os.path.join(out_debug_static_raw, viewpoint_cam.image_name + ".png"),
                        (torch.clamp(
                            (image_second_to_show * light_var.clone().detach().cpu())[0].clone().detach().cpu(), 0,
                            1).permute(1, 2,
                                       0).numpy() * 255).astype(
                            np.uint8)[:, :, ::-1])

                    os.makedirs(out_debug_depth_dir, exist_ok=True)

                    fig, ax = plt.subplots(2, 5, figsize=(30, 12))
                    plt.rcParams['font.family'] = "sans-serif"

                    ax[0, 0].imshow(gt_image_tensor.clone().detach().cpu()[0].permute(1, 2, 0).numpy())
                    ax[0, 0].set_title("Ground Truth")
                    ax[1, 0].imshow(image_tensor.clone().detach().cpu()[0].permute(1, 2, 0).numpy())
                    ax[1, 0].set_title("Full Predicted")
                    # render_pkg_temp = render_egopers(viewpoint_cam, foreground_gaussians, pipeline_config, background, stage=stage,
                    #                        cam_type=scene.dataset_type)
                    ax[0, 1].imshow(image_tensor_first[0].clone().detach().cpu().permute(1, 2, 0).numpy())
                    ax[0, 1].set_title("Dynamic Raw")

                    ax[1, 1].imshow(image_dy[0].clone().detach().cpu().permute(1, 2, 0).numpy())
                    ax[1, 1].set_title("Dynamic Final")

                    # render_pkg_temp = render(viewpoint_cam, background_gaussians, pipeline_config, background, stage=stage,
                    #                        cam_type=scene.dataset_type)
                    ax[0, 2].imshow(image_second_to_show[0].permute(1, 2, 0).numpy())
                    ax[0, 2].set_title("Static Raw")
                    ax[1, 2].imshow(image_sta[0].clone().detach().cpu().permute(1, 2, 0).numpy())
                    ax[1, 2].set_title("Static Final")
                    # ax[0, 1].imshow(np.rot90(depth_array, k=1, axes=(1, 0)), cmap='jet', vmin=0, vmax=10)

                    pos = ax[0, 3].imshow(
                        (motion_masks[0, 2:3, :, :] * pixel_valid_mask[0]).clone().detach().cpu().permute(1, 2,
                                                                                                          0).numpy(),
                        cmap='jet', vmin=0, vmax=1)
                    ax[0, 3].set_title("Dynamic Prob")
                    # ax[1,3].imshow((motion_masks[0, 2:3, :, :].clone().detach().cpu().permute(1, 2, 0).numpy()>vis_thresh), cmap='jet', vmin=0, vmax=1)
                    # ax[1,3].set_title("Valid Mask")
                    ax[1, 3].imshow(((motion_masks[0, 0:1, :,
                                      :] * pixel_valid_mask[0]).clone().detach().cpu().permute(1, 2, 0).numpy()),
                                    cmap='jet',
                                    vmin=0, vmax=1)
                    ax[1, 3].set_title("Brightness Control")
                    ax[0, 4].imshow(image_tensor_second[0].clone().detach().cpu().permute(1, 2, 0).numpy())
                    ax[0, 4].set_title("Brightness Controlled Static")
                    ax[1, 4].imshow(
                        ((mask_comp_first[0] * pixel_valid_mask[0])).clone().detach().cpu().permute(1, 2, 0).numpy(),
                        cmap='jet', vmin=0, vmax=1)
                    ax[1, 4].set_title("Full Probabilistic Mask")

                    # plt.savefig(out_debug_depth_dir + f"/{iteration}.jpg"
                    if which_type == 2:
                        plt.savefig(
                            os.path.join(out_debug_depth_dir, viewpoint_cam.image_name + "_t.jpg"))
                    elif which_type == 1:
                        plt.savefig(
                            os.path.join(out_debug_depth_dir, viewpoint_cam.image_name + "_v.jpg"))
                    else:
                        plt.savefig(
                            os.path.join(out_debug_depth_dir, viewpoint_cam.image_name + ".jpg"))
                    plt.close()

    #####


def training(dataset, hypernetwork_config, optimization_params, pipeline_config, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint,
             debug_from, expname):
    """Entry-point that prepares data structures and launches the two-stage training."""
    tb_writer = prepare_output_and_logger(expname)
    dataset.sh_degree = 3
    foreground_gaussians = GaussianModel_dynamic(dataset.sh_degree, hypernetwork_config)
    dataset.sh_degree = 3
    background_gaussians = GaussianModel(dataset.sh_degree)
    # dataset.sh_degree = 3
    # gaussians_third = GaussianModel(dataset.sh_degree, hypernetwork_config)
    dataset.model_path = args.model_path
    timer = Timer()

    import shutil
    os.makedirs(os.path.join(optimization_params.saving_folder, expname), exist_ok=True)
    shutil.copyfile('./output/' + expname + '/cfg_args',
                    os.path.join(optimization_params.saving_folder, expname, 'cfg_args') )

    scene = Scene2gs_mixed(dataset, foreground_gaussians, load_coarse=None, gaussians_second=background_gaussians)
    foreground_gaussians.max_radii2D = torch.zeros_like(foreground_gaussians.max_radii2D).cuda()
    timer.start()
    scene_reconstruction_degauss(dataset, optimization_params, hypernetwork_config, pipeline_config, testing_iterations, saving_iterations,
                                       checkpoint_iterations, checkpoint, debug_from,
                                       foreground_gaussians, scene, "coarse", tb_writer, optimization_params.coarse_iterations, timer,
                                       background_gaussians=background_gaussians, expname=expname)

    ##### ohhh twice we run this yeah
    scene_reconstruction_degauss(dataset, optimization_params, hypernetwork_config, pipeline_config, testing_iterations, saving_iterations,
                                       checkpoint_iterations, checkpoint, debug_from,
                                       foreground_gaussians, scene, "fine", tb_writer, optimization_params.iterations, timer,
                                       background_gaussians=background_gaussians, expname=expname)

    from distutils.dir_util import copy_tree
    copy_tree("./output/" + expname + "/point_cloud",
              os.path.join(optimization_params.saving_folder, expname,'point_cloud'))
def prepare_output_and_logger(expname):
    """Create run folder and optionally a TensorBoard writer.

    Parameters
    ----------
    expname : str
        Human-readable experiment identifier. The folder `./output/<expname>`
        will be created and the parsed command-line arguments written to
        `cfg_args` for reproducibility.
    """
    if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def setup_seed(seed):
    """Deterministic behaviour across PyTorch / NumPy / random for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[19999, 39999, 79999, 99999, 119999])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--configs", type=str, default="")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams

        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname)
    
    print("\nTraining complete.")
