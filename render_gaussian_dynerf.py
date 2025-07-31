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
import torch.nn as nn
import time as my_time
import cv2
import numpy as np
import random
import os, sys
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss, entropy_loss, l1_loss_with_mask, structural_ssim, \
    ssim_raw, EdgeAwareTV

from gaussian_renderer import render_background, render_foreground, render_mask, render_foreground_with_mask

import sys
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

to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False




# def get_edge_mask(image):
#     # Define Sobel kernels
#     sobel_x = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).unsqueeze(0).unsqueeze(0).to(image.device)
#     sobel_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).unsqueeze(0).unsqueeze(0).to(image.device)
#
#     edge_masks = []
#     for c in range(image.size(1)):  # Apply Sobel filter to each channel separately
#         channel = image[:, c:c + 1, :, :]
#         grad_x = F.conv2d(channel, sobel_x, padding=1)
#         grad_y = F.conv2d(channel, sobel_y, padding=1)
#         grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)
#         edge_masks.append(grad_mag)
#
#     # Combine edge masks from all channels
#     edge_mask = torch.max(torch.stack(edge_masks, dim=1), dim=1)[0]
#
#     # Create binary edge mask (thresholding the gradient magnitude)
#     edge_mask = edge_mask > 0.1
#     edge_mask_np = edge_mask.cpu().numpy()
#     from scipy.ndimage import binary_dilation
#     for i in range(edge_mask_np.shape[0]):
#         edge_mask_np[i] = binary_dilation(edge_mask_np[i], iterations=dilation_iterations)
#
#     return torch.from_numpy(edge_mask_np).to(image.device).float()
#     # Example threshold, adjust as needed
#
#     return edge_mask.float()


def norm_depth(depth_i):
    depth_max = depth_i.max()
    depth_min = depth_i.min()
    return ((depth_i - depth_min) / (depth_max - depth_min))


class CustomActivation(nn.Module):
    def __init__(self):
        super(CustomActivation, self).__init__()

    def forward(self, x):
        output = torch.zeros_like(x)
        mask1 = (x <= 0.75)
        mask2 = (x > 0.75)

        # Linear part for x in [0, 0.75]
        output[mask1] = x[mask1]

        # Linear transformation for x in (0.75, 1]
        output[mask2] = 0.75 + (x[mask2] - 0.75) * ((10 - 0.75) / (1 - 0.75))

        return output





def scene_reconstruction_egogs_pos_neg(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                                       checkpoint_iterations, checkpoint, debug_from,
                                       gaussians, scene, stage, tb_writer, train_iter, timer, gaussians_second=None,
                                       expname='debug_2gs'):
    first_iter = 0


    bg_color = [0, 0, 0]  # sif dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None

    final_iter = train_iter

    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    # lpips_model = lpips.LPIPS(net="alex").cuda()
    use_test = True
    if use_test:
        test_cams = scene.getTestCameras()
    else:
        test_cams = scene.getVideoCameras()

    #
    batch_size = opt.batch_size
    print("data loading done")
    if opt.dataloader:
        viewpoint_stack = scene.getTrainCameras()
        if opt.custom_sampler is not None:
            sampler = FineSampler(viewpoint_stack)
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size, sampler=sampler, num_workers=16,
                                                collate_fn=list)
            random_loader = False
        else:
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size, shuffle=True, num_workers=16,
                                                collate_fn=list)
            random_loader = True
        loader = iter(viewpoint_stack_loader)

    # dynerf, zerostamp_init
    # breakpoint()

    ####### mask set up
    import cv2
    import glob


    ############# eval after training
    if stage == "fine":
        all_times = []

        batch_size = 1
        viewpoint_stack_index = list(range( len(test_cams)))
        # if not viewpoint_stack and not opt.dataloader:
        # dnerf's branch
        del viewpoint_stack
        # data_list = [0] * len(train_cams) + [1] * len(val_cams) + [2] * len(test_cams)
        # viewpoint_stack = [i for i in train_cams] + [i for i in test_cams]+ [i for i in test_cams]
        data_list = [2] * len(test_cams)
        viewpoint_stack = [i for i in test_cams]
        for iteration in range(0, len(viewpoint_stack)):

            iter_start.record()

            # Every 1000 its we increase the levels of SH up to a maximum degree

            # dynerf's branch

            if True:
                idx = 0
                viewpoint_cams = []

                while idx < batch_size:
                    try:
                        viewpoint_cam_idx = viewpoint_stack_index.pop(0)
                    except:
                        return 0
                    viewpoint_cam = viewpoint_stack[viewpoint_cam_idx]
                    which_type = data_list[viewpoint_cam_idx]
                    # viewpoint_cam = viewpoint_stack.pop(iteration)
                    # if not viewpoint_stack:
                    #     viewpoint_stack = temp_list.copy()
                    viewpoint_cams.append(viewpoint_cam)
                    idx += 1
                if len(viewpoint_cams) == 0:
                    continue
            # print(len(viewpoint_cams))
            # breakpoint()
            # Render
            light_field_exist = True
            with torch.no_grad():
                # if (iteration - 1) == debug_from:
                #     pipe.debug = False
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
                    render_pkg_dynamic_pers, render_pkg_motion = render_foreground_with_mask(viewpoint_cam, gaussians, pipe, background, stage=stage,
                                                             cam_type=scene.dataset_type)
                    render_pkg_second = render_foreground(viewpoint_cam, gaussians_second, pipe, background, stage='coarse',
                                               cam_type=scene.dataset_type)

                    image, viewspace_point_tensor, visibility_filter, radii = render_pkg_dynamic_pers["render"], \
                        render_pkg_dynamic_pers[
                            "viewspace_points"], render_pkg_dynamic_pers["visibility_filter"], render_pkg_dynamic_pers[
                        "radii"]


                    images.append(image.unsqueeze(0))

                    gt_image = viewpoint_cam.original_image.float().cuda() / 255

                    # render_pkg_second = render(viewpoint_cam, gaussians, pipe, background, stage=stage, cam_type=scene.dataset_type)
                    ### no deformation for second model

                    image_second = render_pkg_second["render"]


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

                radii = torch.cat(radii_list, 0).max(dim=0).values
                motion_masks = torch.cat(motion_masks, 0)
                visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
                image_tensor_first = torch.cat(images, 0)
                gt_image_tensor = torch.cat(gt_images, 0)
                image_tensor_second = torch.cat(images_second, 0)
                radii_second = torch.cat(radii_list_second, 0).max(dim=0).values
                visibility_filter_second = torch.cat(visibility_filter_list_second).any(dim=0)
                pixel_valid_mask = torch.ones_like(gt_image_tensor)[0, 0].unsqueeze(0).unsqueeze(0).float().cuda()

                motion_pro_first = motion_masks[:, 2:3, :, :] + 1e-6
                motion_pro_second = motion_masks[:, 1:2, :, :] + 1e-6

                if True:

                    image_tensor_second = image_tensor_second * pixel_valid_mask
                    gt_image_tensor = gt_image_tensor
                    image_tensor_first = image_tensor_first * pixel_valid_mask

                    activation_light = CustomActivation()
                    light_var = 0.5 + activation_light(motion_masks[:, 0:1, :, :].repeat(1, 3, 1, 1))

                    white_thresh = 0.9


                    if light_field_exist:
                        image_second_to_show = image_tensor_second.clone().detach().cpu()
                        image_second_to_show_g = image_tensor_second.clone().detach()
                        image_tensor_second = image_tensor_second * light_var
                        image_tensor_second = torch.clamp(image_tensor_second, 0, 1 - 1e-9)
                        image_tensor_first = image_tensor_first
                        image_tensor_first = torch.clamp(image_tensor_first, 0, 1 - 1e-9)

                    else:
                        image_second_to_show = image_tensor_second.clone().detach().cpu()
                        image_second_to_show_g = image_tensor_second.clone().detach()

                    distri = motion_pro_first + motion_pro_second
                    motion_masks_first = motion_pro_first / distri
                    motion_masks_second = motion_pro_second / distri

                    # motion_masks_first = (motion_masks_first >= 0.51) * motion_masks_first
                    # motion_masks_second = 1 - motion_masks_first

                    mask_comp_first = motion_masks_first
                    mask_comp_second = motion_masks_second

                    image_dy = image_tensor_first * motion_masks_first
                    image_sta = image_tensor_second * motion_masks_second
                    image_tensor = image_dy + image_sta

                if iteration == opt.iterations:
                    progress_bar.close()

                if iteration % 1 == 0:
                    import matplotlib.pyplot as plt
                    out_debug_depth_dir = os.path.join(opt.saving_folder, expname,
                                                       'train_cams')

                    out_debug_gt = os.path.join(opt.saving_folder, expname, 'gt')
                    os.makedirs(out_debug_gt, exist_ok=True)
                    cv2.imwrite(os.path.join(out_debug_gt, viewpoint_cam.image_name + ".png"),
                                (torch.clamp(gt_image_tensor[0].clone().detach().cpu(), 0, 1).permute(1, 2,
                                                                                                      0).numpy() * 255).astype(
                                    np.uint8)[:, :, ::-1])
                    out_debug_full_pred = os.path.join(opt.saving_folder, expname,
                                                       'full_pred')
                    os.makedirs(out_debug_full_pred, exist_ok=True)
                    cv2.imwrite(
                        os.path.join(out_debug_full_pred, viewpoint_cam.image_name + ".png"),
                        (torch.clamp(image_tensor[0].clone().detach().cpu(), 0, 1).permute(1, 2,
                                                                                           0).numpy() * 255).astype(
                            np.uint8)[:, :, ::-1])
                    out_debug_mask = os.path.join(opt.saving_folder, expname,
                                                  'mask_comp')
                    os.makedirs(out_debug_mask, exist_ok=True)
                    feature_out_path = os.path.join(out_debug_mask, viewpoint_cam.image_name + ".npy")
                    with open(feature_out_path, "wb") as fout:
                        np.save(fout, ((mask_comp_first[0] * pixel_valid_mask[0])).clone().detach().cpu().permute(1, 2,
                                                                                                                  0).numpy())

                    out_debug_static_raw = os.path.join(opt.saving_folder, expname,
                                                        'static_raw')
                    os.makedirs(out_debug_static_raw, exist_ok=True)
                    cv2.imwrite(
                        os.path.join(out_debug_static_raw, viewpoint_cam.image_name + ".png"),
                        (torch.clamp(image_second_to_show[0].clone().detach().cpu(), 0, 1).permute(1, 2,
                                                                                                   0).numpy() * 255).astype(
                            np.uint8)[:, :, ::-1])

                    os.makedirs(out_debug_depth_dir, exist_ok=True)

                    out_debug_static_raw = os.path.join(opt.saving_folder, expname,
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
                    # render_pkg_temp = render_egopers(viewpoint_cam, gaussians, pipe, background, stage=stage,
                    #                        cam_type=scene.dataset_type)
                    ax[0, 1].imshow(image_tensor_first[0].clone().detach().cpu().permute(1, 2, 0).numpy())
                    ax[0, 1].set_title("Dynamic Raw  Part")

                    ax[1, 1].imshow(image_dy[0].clone().detach().cpu().permute(1, 2, 0).numpy())
                    ax[1, 1].set_title("Dynamic Filtered Part")

                    # render_pkg_temp = render(viewpoint_cam, gaussians_second, pipe, background, stage=stage,
                    #                        cam_type=scene.dataset_type)
                    ax[0, 2].imshow(image_second_to_show[0].permute(1, 2, 0).numpy())
                    ax[0, 2].set_title("Static Raw Part")
                    ax[1, 2].imshow(image_sta[0].clone().detach().cpu().permute(1, 2, 0).numpy())
                    ax[1, 2].set_title("Static Filtered Part")
                    # ax[0, 1].imshow(np.rot90(depth_array, k=1, axes=(1, 0)), cmap='jet', vmin=0, vmax=10)

                    pos = ax[0, 3].imshow(
                        (motion_masks[0, 2:3, :, :] * pixel_valid_mask[0]).clone().detach().cpu().permute(1, 2,
                                                                                                          0).numpy(),
                        cmap='jet', vmin=0, vmax=1)
                    ax[0, 3].set_title("Predicted Mask")
                    # ax[1,3].imshow((motion_masks[0, 2:3, :, :].clone().detach().cpu().permute(1, 2, 0).numpy()>vis_thresh), cmap='jet', vmin=0, vmax=1)
                    # ax[1,3].set_title("Valid Mask")
                    ax[1, 3].imshow(((motion_masks[0, 0:1, :,
                                      :] * pixel_valid_mask[0]).clone().detach().cpu().permute(1, 2, 0).numpy()),
                                    cmap='jet',
                                    vmin=0, vmax=1)
                    ax[1, 3].set_title("Light Variation")
                    ax[0, 4].imshow(image_tensor_second[0].clone().detach().cpu().permute(1, 2, 0).numpy())
                    ax[0, 4].set_title("Static Raw with light Part")
                    ax[1, 4].imshow(
                        ((mask_comp_first[0] * pixel_valid_mask[0])).clone().detach().cpu().permute(1, 2, 0).numpy(),
                        cmap='jet', vmin=0, vmax=1)
                    ax[1, 4].set_title("Dynamic Mask")

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



def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint,
             debug_from, expname):
    # first_iter = 0
    dataset.sh_degree = 3
    gaussians = GaussianModel_dynamic(dataset.sh_degree, hyper)
    dataset.sh_degree = 3
    gaussians_second = GaussianModel(dataset.sh_degree)
    # dataset.sh_degree = 3
    # gaussians_third = GaussianModel(dataset.sh_degree, hyper)
    new_check = args.render_checkpoint
    dataset.model_path = args.model_path
    timer = Timer()


    # new_check= '/media/ray/data_volume/dynerf_run_debug/trytry7buhao/flame_salmon_sparse'

    # scene = Scene(dataset, gaussians, load_coarse=None)
    # scene = Scene(dataset, gaussians, load_coarse=None)
    # scene = Scene2gs(dataset, gaussians, load_coarse=None, gaussians_second=gaussians_second)
    scene = Scene2gs_mixed(dataset, gaussians, load_coarse=None, gaussians_second=gaussians_second,load_iteration=new_check)
    gaussians.max_radii2D = torch.zeros_like(gaussians.max_radii2D).cuda()
    timer.start()

    scene_reconstruction_egogs_pos_neg(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                                       checkpoint_iterations, checkpoint, debug_from,
                                       gaussians, scene, "fine", None, opt.iterations, timer,
                                       gaussians_second=gaussians_second, expname=expname)

    # from distutils.dir_util import copy_tree
    # copy_tree("./output/" + expname + "/point_cloud",
    #           "/media/ray/data_volume/dynerf_run_debug/" + expname + "/point_cloud")




def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False


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
    parser.add_argument("--render_checkpoint", type=str, default=None)
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
    # network_gui.init(args.ip, args.port)
    # torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations,
             args.save_iterations, args.checkpoint_iterations, args.render_checkpoint, args.debug_from, args.expname)

    # All done
    print("\nTraining complete.")
