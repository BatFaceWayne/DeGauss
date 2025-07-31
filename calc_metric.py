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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from pytorch_msssim import ms_ssim
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def readImages(renders_dir, gt_dir, train_cams, mode='val'):
    renders = []
    gts = []
    image_names = []
    list_of_cams = os.listdir(train_cams)
    for fname in os.listdir(renders_dir):
        extention_flag = '_t.' if mode == 'test' else '_v.'
        if fname.split('.')[0] + extention_flag + 'png' not in list_of_cams and fname.split('.')[
            0] + extention_flag + 'jpg' not in list_of_cams:
            continue
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        # renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :])
        # gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :])
        renders.append(render)
        gts.append(gt)
        image_names.append(fname)

    return renders, gts, image_names


def evaluate(mode='test', expname='debug_2gs', data_path='./test/'):
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in ["this_metric"]:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(os.path.join(data_path , expname))

            for method in ["decomp_gs"]:
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir
                gt_dir = method_dir / "gt"
                renders_dir = method_dir / "full_pred"
                train_cams = method_dir / "train_cams"
                # renders_dir = method_dir / "static_raw"
                renders, gts, image_names = readImages(renders_dir, gt_dir, train_cams, mode=mode)

                ssims = []
                psnrs = []
                lpipss = []
                lpipsa = []
                ms_ssims = []
                Dssims = []
                my_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
                my_psnr = PeakSignalNoiseRatio(data_range=1.0).cuda()
                my_lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).cuda()
                lpipsvgg = []
                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):

                    this_render = tf.to_tensor(renders[idx]).unsqueeze(0)[:, :3, :, :].cuda()
                    this_gt = tf.to_tensor(gts[idx]).unsqueeze(0)[:, :3, :, :].cuda()

                    ssims.append(my_ssim(this_render, this_gt))
                    psnrs.append(my_psnr(this_render, this_gt))
                    lpipss.append(my_lpips(this_render, this_gt))
                    try:
                        ms_ssims.append(ms_ssim(this_render, this_gt, data_range=1, size_average=True))
                    except:
                        ms_ssims.append(1.0)
                    lpipsa.append(lpips(this_render, this_gt, net_type='alex'))

                    lpipsvgg.append(lpips(this_render, this_gt, net_type='vgg'))
                    Dssims.append((1 - ms_ssims[-1]) / 2)

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                     "PSNR": torch.tensor(psnrs).mean().item(),
                                                     "LPIPS-alex normalized": torch.tensor(lpipss).mean().item(),
                                                     "LPIPS-alex": torch.tensor(lpipsa).mean().item(),
                                                     "LPIPS-vgg": torch.tensor(lpipsvgg).mean().item(),
                                                     "MS-SSIM": torch.tensor(ms_ssims).mean().item(),
                                                     "D-SSIM": torch.tensor(Dssims).mean().item()},

                                                    )
                per_view_dict[scene_dir][method].update(
                    {"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                     "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                     "LPIPS-alex normalized": {name: lp for lp, name in
                                               zip(torch.tensor(lpipss).tolist(), image_names)},
                     "LPIPS-alex": {name: lp for lp, name in zip(torch.tensor(lpipsa).tolist(), image_names)},
                     "LPIPS-vgg": {name: lp for lp, name in zip(torch.tensor(lpipsvgg).tolist(), image_names)},
                     "MS-SSIM": {name: lp for lp, name in zip(torch.tensor(ms_ssims).tolist(), image_names)},
                     "D-SSIM": {name: lp for lp, name in zip(torch.tensor(Dssims).tolist(), image_names)},

                     }
                )

            if mode == 'val':
                with open(str(test_dir) + "/results_val.json", 'w') as fp:
                    json.dump(full_dict[scene_dir], fp, indent=True)
                with open(str(test_dir) + "/per_view_val.json", 'w') as fp:
                    json.dump(per_view_dict[scene_dir], fp, indent=True)
            else:
                with open(str(test_dir) + "/results_test.json", 'w') as fp:
                    json.dump(full_dict[scene_dir], fp, indent=True)
                with open(str(test_dir) + "/per_view_test.json", 'w') as fp:
                    json.dump(per_view_dict[scene_dir], fp, indent=True)

        except Exception as e:

            print("Unable to compute metrics for model", scene_dir)
            raise e


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--data_path', '-d', required=True, nargs="+", type=str,
                        default='./test/')

    parser.add_argument('--scene_name', '-s', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    expname = args.scene_name[0]
    data_path = args.data_path[0]
    evaluate(mode='val', expname=expname, data_path=data_path)

    evaluate(mode='test', expname=expname,  data_path=data_path)
