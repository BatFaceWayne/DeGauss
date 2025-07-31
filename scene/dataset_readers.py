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
# -----------------------------------------------------------------------------
# Dataset Readers and Scene Builders
# -----------------------------------------------------------------------------
# Utility functions for loading Colmap/Blender/HyperNeRF cameras and constructing
# SceneInfo objects. This is a high-level IO layer and should remain free of any
# heavy math or rendering code.

import os
import os
import sys
from PIL import Image,ImageEnhance
from scene.cameras import Camera
import mediapy as media

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from scene.hyper_loader import Load_hyper_data, format_hyper_data
import torchvision.transforms as transforms
import copy
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import torch
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.general_utils import PILtoTorch
from tqdm import tqdm
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time : float
    mask: np.array
    SD_feature: np.array
   
class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    video_cameras: list
    nerf_normalization: dict
    ply_path: str
    maxtime: int
    point_cloud_second: BasicPointCloud = None
    point_cloud_third: BasicPointCloud = None
    val_cameras: list = None

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center
    # breakpoint()
    return {"translate": translate, "radius": radius, "cam_centers":cam_centers}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    keys_list = list(cam_extrinsics.keys())
    keys_list = sorted(keys_list)

    import json

    json_base = '/media/ray/data_volume/aria_data/epic_field/split'
    json_train_path = os.path.join(json_base, 'train.json')
    json_test_path = os.path.join(json_base, 'test.json')
    json_val_path = os.path.join(json_base, 'val.json')

    json_train = json.load(open(json_train_path, 'r'))
    json_test = json.load(open(json_test_path, 'r'))
    json_val = json.load(open(json_val_path, 'r'))
    base_experiment = images_folder.split('/')[-2]
    train_all_list = json_train[base_experiment]
    test_all_list = []
    for k,v in enumerate(json_test[base_experiment]):
        test_all_list +=json_test[base_experiment][v]
    val_all_list = []
    for k,v in enumerate(json_val[base_experiment]):
        val_all_list +=json_val[base_experiment][v]

    all_frame_list = train_all_list + test_all_list + val_all_list
    all_frame_list_num = [float(os.path.basename(x).split('.')[0].split('_')[-1]) for x in all_frame_list]
    np_list = np.array(all_frame_list_num)
    frame_max = np_list.max()
    frame_min = np_list.min()
    for idx in range(len(keys_list)):
        key = keys_list[idx]

        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        if extr.name not in all_frame_list:
            continue

        extr.name
        # 98824
        if float(extr.name.split('.')[0].split('_')[-1]) >90000 :
            continue
        if float(extr.name.split('.')[0].split('_')[-1]) >60000 or float(extr.name.split('.')[0].split('_')[-1]) < 50000:
            continue
        # if float(extr.name.split('.')[0].split('_')[-1]) >60000 or float(extr.name.split('.')[0].split('_')[-1]) < 55000 or int(extr.name.split('.')[0].split('_')[-1]) %2 ==0:
        #     continue
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        downscale = 1
        if intr.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]

        image = Image.open(image_path)
        if downscale!=1:
            image = image.resize((width//downscale, height//downscale), Image.LANCZOS)
            focal_length_x = intr.params[0]//downscale
            focal_length_y = intr.params[1]//downscale
            width = width//downscale
            height = height//downscale
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        enhance_brightness_flag = False
        if enhance_brightness_flag:
            image = ImageEnhance.Brightness(image)
            image = image.enhance(1.5)
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        use_feature = False
        if not use_feature:
            feature_out = np.zeros(1)
        if use_feature:
            feature_out_path = os.path.join(os.path.dirname(os.path.dirname(image_path)), 'SD_out', os.path.basename(image_path).split('.')[0] + '.npy')
            if os.path.exists(feature_out_path):
                feature_out = np.load(feature_out_path)
            else:
                print("no clustered feature, processing now.")
                SD_feature_path = os.path.join(os.path.dirname(os.path.dirname(image_path)), 'SD', os.path.basename(image_path).split('.')[0] + '.npy')

                SD_feature = np.load(SD_feature_path)
                if True:
                    ft_flat = np.transpose(SD_feature.reshape((1280, 44 * 44)), (1, 0))
                    x = np.linspace(0, 1, 44)
                    y = np.linspace(0, 1, 44)
                    xv, yv = np.meshgrid(x, y)
                    indxy = np.reshape(np.stack([xv, yv], axis=-1), (44 * 44, 2))
                    knn_graph = kneighbors_graph(indxy, 8, include_self=False)
                    model = AgglomerativeClustering(
                        linkage="ward", connectivity=knn_graph, n_clusters=100
                    )
                    model.fit(ft_flat)
                    feature_out = np.array(
                        [model.labels_ == i for i in range(model.n_clusters)],
                        dtype=np.float32,
                    ).reshape((model.n_clusters, 44, 44))
                    os.makedirs(os.path.dirname(feature_out_path), exist_ok=True)
                    with open(feature_out_path, "wb") as fout:
                        np.save(fout, feature_out)

        # image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()/255

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                          image_path=image_path, image_name=image_name, width=width, height=height,
                          time = float(  (float(extr.name.split('.')[0].split('_')[-1]) - frame_min)/(frame_max - frame_min) ), mask=None, SD_feature=feature_out) # default by monocular settings.
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readColmapCameras_onthego_dist(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    keys_list = list(cam_extrinsics.keys())
    keys_list = sorted(keys_list)

    import json

    json_base = os.path.join(os.path.dirname(images_folder), 'split.json')

    # json_split = json.load(open(json_base, 'r'))

    # train_all_list = json_split['clutter']
    # test_all_list = json_split['extra']
    # val_all_list = json_split['extra']
    #
    # source_json  = os.path.join(os.path.dirname(images_folder),'transforms.json')
    # source_list = json.load(open(source_json, 'r'))
    # create_map = {i : v['file_path'].split('/')[-1] for i ,v in enumerate(source_list['frames'])}
    # create_map_reverse = {v['file_path'].split('/')[-1] : i for i ,v in enumerate(source_list['frames'])}

    all_images = [float(val.name.split('.')[0].split('extra')[-1].split('clutter')[-1].split('clean')[-1]) for key, val
                  in cam_extrinsics.items()]
    for idx in range(len(keys_list)):
        key = keys_list[idx]

        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]

        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        downscale = 8
        all_names_folder = images_folder.split('/')
        any_match_folder_name = [all_names_folder[iitte] == 'patio_new' for iitte in range(len(all_names_folder))]
        if any(any_match_folder_name):
            downscale = 4
        if intr.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        distort_params = intr.params[4:]
        image = Image.open(image_path)
        if downscale != 1:
            image = media.resize_image(image, (height// downscale, width// downscale))

            # image = image.resize((width // downscale, height // downscale), Image.LANCZOS)
            focal_length_x = intr.params[0] / downscale
            focal_length_y = intr.params[1] / downscale
            K = np.array(
                [[focal_length_x, 0, width // downscale / 2], [0, focal_length_y, height // downscale / 2], [0, 0, 1]])
            import cv2
            K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                K, np.array(distort_params), (width // downscale, height // downscale), 0
            )
            mapx, mapy = cv2.initUndistortRectifyMap(
                K, np.array(distort_params), None, K_undist, (width // downscale, height // downscale), cv2.CV_32FC1
            )
            image = cv2.remap(np.array(image), mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = roi_undist
            image = image[y: y + h, x: x + w]

            # width = width//downscale
            # height = height//downscale
            width = w  # width//downscale
            height = h  # height//downscale
            focal_length_y = K_undist[1, 1]
            focal_length_x = K_undist[0, 0]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        enhance_brightness_flag = False
        if enhance_brightness_flag:
            image = ImageEnhance.Brightness(image)
            image = image.enhance(1.5)
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        use_feature = False
        if not use_feature:
            feature_out = np.zeros(1)
        if use_feature:
            feature_out_path = os.path.join(os.path.dirname(os.path.dirname(image_path)), 'SD_out',
                                            os.path.basename(image_path).split('.')[0] + '.npy')
            if os.path.exists(feature_out_path):
                feature_out = np.load(feature_out_path)
            else:
                print("no clustered feature, processing now.")
                SD_feature_path = os.path.join(os.path.dirname(os.path.dirname(image_path)), 'SD',
                                               os.path.basename(image_path).split('.')[0] + '.npy')

                SD_feature = np.load(SD_feature_path)
                if True:
                    ft_flat = np.transpose(SD_feature.reshape((1280, 44 * 44)), (1, 0))
                    x = np.linspace(0, 1, 44)
                    y = np.linspace(0, 1, 44)
                    xv, yv = np.meshgrid(x, y)
                    indxy = np.reshape(np.stack([xv, yv], axis=-1), (44 * 44, 2))
                    knn_graph = kneighbors_graph(indxy, 8, include_self=False)
                    model = AgglomerativeClustering(
                        linkage="ward", connectivity=knn_graph, n_clusters=100
                    )
                    model.fit(ft_flat)
                    feature_out = np.array(
                        [model.labels_ == i for i in range(model.n_clusters)],
                        dtype=np.float32,
                    ).reshape((model.n_clusters, 44, 44))
                    os.makedirs(os.path.dirname(feature_out_path), exist_ok=True)
                    with open(feature_out_path, "wb") as fout:
                        np.save(fout, feature_out)
        this_time = float(image_name.split('.')[0].split('extra')[-1].split('clutter')[-1].split('clean')[-1]) / max(
            all_images)

        # image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()/255

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              time=this_time, mask=None, SD_feature=feature_out)  # default by monocular settings.
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos
def readColmapCameras_deblur(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    keys_list = list(cam_extrinsics.keys())
    keys_list = sorted(keys_list)
    all_time = []
    for idx in range(len(keys_list)):
        all_time.append(float(cam_extrinsics[keys_list[idx]].name.split('.')[0].split('_')[-1]) )
    for idx in range(len(keys_list)):
        key = keys_list[idx]

        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        time_of_this = float(extr.name.split('.')[0].split('_')[-1])

        # if float(extr.name.split('.')[0].split('_')[-1]) >60000 or float(extr.name.split('.')[0].split('_')[-1]) < 55000 or int(extr.name.split('.')[0].split('_')[-1]) %2 ==0:
        #     continue
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        downscale = 1
        if intr.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]

        image = Image.open(image_path)
        if downscale!=1:
            image = image.resize((width//downscale, height//downscale), Image.LANCZOS)
            focal_length_x = intr.params[0]//downscale
            focal_length_y = intr.params[1]//downscale
            width = width//downscale
            height = height//downscale
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        enhance_brightness_flag = False
        if enhance_brightness_flag:
            image = ImageEnhance.Brightness(image)
            image = image.enhance(1.5)
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        use_feature = False
        if not use_feature:
            feature_out = np.zeros(1)
        if use_feature:
            feature_out_path = os.path.join(os.path.dirname(os.path.dirname(image_path)), 'SD_out', os.path.basename(image_path).split('.')[0] + '.npy')
            if os.path.exists(feature_out_path):
                feature_out = np.load(feature_out_path)
            else:
                print("no clustered feature, processing now.")
                SD_feature_path = os.path.join(os.path.dirname(os.path.dirname(image_path)), 'SD', os.path.basename(image_path).split('.')[0] + '.npy')

                SD_feature = np.load(SD_feature_path)
                if True:
                    ft_flat = np.transpose(SD_feature.reshape((1280, 44 * 44)), (1, 0))
                    x = np.linspace(0, 1, 44)
                    y = np.linspace(0, 1, 44)
                    xv, yv = np.meshgrid(x, y)
                    indxy = np.reshape(np.stack([xv, yv], axis=-1), (44 * 44, 2))
                    knn_graph = kneighbors_graph(indxy, 8, include_self=False)
                    model = AgglomerativeClustering(
                        linkage="ward", connectivity=knn_graph, n_clusters=100
                    )
                    model.fit(ft_flat)
                    feature_out = np.array(
                        [model.labels_ == i for i in range(model.n_clusters)],
                        dtype=np.float32,
                    ).reshape((model.n_clusters, 44, 44))
                    os.makedirs(os.path.dirname(feature_out_path), exist_ok=True)
                    with open(feature_out_path, "wb") as fout:
                        np.save(fout, feature_out)

        # image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()/255

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                          image_path=image_path, image_name=image_name, width=width, height=height,
                          time = float(  time_of_this / max(all_time) ), mask=None, SD_feature=feature_out) # default by monocular settings.
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos
def readColmapCameras_hypernerf(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    keys_list = list(cam_extrinsics.keys())
    keys_list = sorted(keys_list)
    all_time = []
    for idx in range(len(keys_list)):
        all_time.append(float(cam_extrinsics[keys_list[idx]].name.split('.')[0].split('_')[-1]))
    for idx in range(len(keys_list)):
        key = keys_list[idx]

        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        time_of_this = float(extr.name.split('.')[0].split('_')[-1])

        # if float(extr.name.split('.')[0].split('_')[-1]) >60000 or float(extr.name.split('.')[0].split('_')[-1]) < 55000 or int(extr.name.split('.')[0].split('_')[-1]) %2 ==0:
        #     continue
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        downscale = 1
        if intr.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]

        image = Image.open(image_path)

        if downscale != 1:
            image = image.resize((width // downscale, height // downscale), Image.LANCZOS)
            focal_length_x = intr.params[0] / downscale
            focal_length_y = intr.params[1] / downscale
            width = width // downscale
            height = height // downscale
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        enhance_brightness_flag = False
        if enhance_brightness_flag:
            image = ImageEnhance.Brightness(image)
            image = image.enhance(1.5)
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        use_feature = False
        if not use_feature:
            feature_out = np.zeros(1)

        # image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()/255

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              time=float(time_of_this / max(all_time)), mask=None,
                              SD_feature=feature_out)  # default by monocular settings.
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'f4'), ('green', 'f4'), ('blue', 'f4')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    # breakpoint()
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)
def readColmapCameras_aria(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    keys_list = list(cam_extrinsics.keys())
    keys_list = sorted(keys_list)
    wait_flames = 0
    wait_flames_thresh = 50
    # wait 2 seconds for the brightness to adjust.
    all_frames = 0
    all_frames_thresh = 5000
    start_flag = False
    end_flag = False
    start_frame = None #'camera-rgb_404453130987.jpg'
    end_frame = None
    # for idx, key in enumerate(cam_extrinsics):
    for idx in range(len(keys_list)):
        key = keys_list[idx]

        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        if start_frame is not None:
            if start_frame == extr.name:
                start_flag = True
            if end_frame == extr.name:
                end_flag = True
            if not start_flag:
                continue
            if end_flag:
                break
        if wait_flames < wait_flames_thresh:
            wait_flames += 1
            continue

        extr.name
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        downscale = 2
        if intr.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]

        image = Image.open(image_path)
        if downscale!=1:
            image = image.resize((width//downscale, height//downscale), Image.LANCZOS)
            focal_length_x = intr.params[0]/downscale
            focal_length_y = intr.params[1]/downscale
            width = width//downscale
            height = height//downscale
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        enhance_brightness_flag = False
        if enhance_brightness_flag:
            image = ImageEnhance.Brightness(image)
            image = image.enhance(1.5)
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        use_feature = False
        if not use_feature:
            feature_out = np.zeros(1)



        if all_frames < all_frames_thresh:
            all_frames += 1
        else:
            break
        # cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
        #                       image_path=image_path, image_name=image_name, width=width, height=height,
        #                       time=float(all_frames / all_frames_thresh), mask=None)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                          image_path=image_path, image_name=image_name, width=width, height=height,
                          time = float(key/len(cam_extrinsics)), mask=None, SD_feature=feature_out) # default by monocular settings.
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos
def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    use_1_user = False
    useonthego = False
    usephoto= False
    use_hyper = False

    if os.path.exists( os.path.join(path,'global_points.ply')):
        use_1_user = True

    all_images_names = os.listdir(os.path.join(path, reading_dir))
    if any(['clutter' in temp_im for temp_im in all_images_names]):
        useonthego = True

    if os.path.exists(os.path.join(path, 'dataset.json')):
        use_hyper = True


    if use_1_user:
        cam_infos = readColmapCameras_aria(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                      images_folder=os.path.join(path, reading_dir))
        egolifter_style = False
        if egolifter_style:
            thresh_train_val = int(len(cam_infos) * 0.8)
            cam_infos_train_val = [c for idx, c in enumerate(cam_infos) if idx <= thresh_train_val]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx > thresh_train_val]

            train_cam_infos = [c for idx, c in enumerate(cam_infos_train_val) if idx % 5 != 0]
            val_cam_infos = [c for idx, c in enumerate(cam_infos_train_val) if idx % 5 == 0]
        else:

            cam_infos_train_val = cam_infos
            train_cam_infos = [c for idx, c in enumerate(cam_infos_train_val) if idx % 5 != 0]
            val_cam_infos = [c for idx, c in enumerate(cam_infos_train_val) if idx % 5 == 0]
            test_cam_infos = val_cam_infos
        nerf_normalization = getNerfppNorm(train_cam_infos)


        pc_in_aria = os.path.join(path, "global_points.ply")
        ply_path = pc_in_aria
        import open3d as o3d
        pcd_aria = o3d.io.read_point_cloud(pc_in_aria)
        transform_matrix = np.array([[1., 0., 0., 0.],
                                     [0., 0., 1., 0.],
                                     [0., -1., 0., 0.], [0, 0, 0, 1]])
        pcd_aria.transform(np.linalg.inv(transform_matrix))
        xyz = np.array(pcd_aria.points)
        shs = np.random.random(np.shape(xyz))  # / 255.0

        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros(np.shape(xyz)))
        pcd_max = xyz.max(axis=0) #+ 0.1 * nerf_normalization['radius']
        pcd_min = xyz.min(axis=0) #- 0.1 * nerf_normalization['radius']
        num_pts = 10000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        x = np.random.random((num_pts, 1)) * (pcd_max[0] - pcd_min[0]) + pcd_min[0]
        y = np.random.random((num_pts, 1)) * (pcd_max[1] - pcd_min[1]) + pcd_min[1]
        z = np.random.random((num_pts, 1)) * (pcd_max[2] - pcd_min[2]) + pcd_min[2]
        xyz = np.hstack([x, y, z])
        shs = np.random.random((num_pts, 3))  # / 255.0
        pcd_2 = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))


        pcd_3 = None
    elif useonthego:
        cam_infos = readColmapCameras_onthego_dist(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                                   images_folder=os.path.join(path, reading_dir))
        import json

        test_cam_infos = []
        val_cam_infos = []
        train_cam_infos = []
        for idx, c in enumerate(cam_infos):
            if 'extra' in c.image_path.split('/')[-1]:
                test_cam_infos.append(c)
            elif 'clutter' in c.image_path.split('/')[-1]:
                train_cam_infos.append(c)
        val_cam_infos = [test_cam_infos[0]]


        nerf_normalization = getNerfppNorm(train_cam_infos)

        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")

        using_colmap = True
        if using_colmap:

            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)

            storePly(ply_path, xyz, rgb)

            use_dense = False
            if use_dense:
                pc_in_aria = os.path.join(path, "fused.ply")
                import open3d as o3d
                pcd_aria = o3d.io.read_point_cloud(pc_in_aria)

                xyz = np.array(pcd_aria.points)

            xyz_copy_ori = np.array(xyz)
            use_sky_pc = True
            sky_pc = None
            if use_sky_pc:
                center = -nerf_normalization['translate']  # Center of the scene
                sphere_radius = np.sqrt(((xyz_copy_ori - center) ** 2).sum(axis=1)).max() * 2

                # Radius of the scene
                num_points = 100000  # Number of points for the sphere
                phi = np.random.uniform(0, 2 * np.pi, num_points)  # Azimuthal angle
                cos_theta = np.random.uniform(-1, 1, num_points)  # Uniform distribution for theta
                theta = np.arccos(cos_theta)
                x = sphere_radius * np.sin(theta) * np.cos(phi) + center[0]
                y = sphere_radius * np.sin(theta) * np.sin(phi) + center[1]
                z = sphere_radius * np.cos(theta) + center[2]

                # Combine into a point cloud
                sky_pc = np.vstack((x, y, z)).T

            if sky_pc is not None:
                xyz = np.vstack((xyz_copy_ori, sky_pc))


            shs_base = rgb / 255
            shs = np.random.random((xyz.shape[0], 3))  # / 255.0
            shs[:(len(shs_base))] = shs_base


            pcd = BasicPointCloud(points=xyz, colors=(shs), normals=np.zeros(np.shape(xyz)))
            pcd_max = xyz_copy_ori.max(axis=0) + 0.1 * nerf_normalization['radius']
            pcd_min = xyz_copy_ori.min(axis=0) - 0.1 * nerf_normalization['radius']
            num_pts = 30000
            print(f"Generating random point cloud ({num_pts})...")

            # We create random points inside the bounds of the synthetic Blender scenes
            xyz_len = list(range(xyz_copy_ori.shape[0]))
            choice_selected = np.random.choice(xyz_len, min(len(xyz_len) // 10, 100), replace=False)
            x = np.random.random((num_pts, 1)) * (pcd_max[0] - pcd_min[0]) + pcd_min[0]
            y = np.random.random((num_pts, 1)) * (pcd_max[1] - pcd_min[1]) + pcd_min[1]
            z = np.random.random((num_pts, 1)) * (pcd_max[2] - pcd_min[2]) + pcd_min[2]
            xyz_random = np.hstack([x, y, z])
            xyz = xyz_random
            # xyz = np.vstack((xyz_copy_ori[choice_selected, :], xyz_random))
            shs = np.random.random((xyz.shape[0], 3))  # 255.0
            pcd_2 = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((len(xyz), 3)))
            #
            # pcd_2 = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

            pc3_scale = 6

            x = np.random.random((num_pts, 1)) * pc3_scale - pc3_scale / 2
            y = np.random.random((num_pts, 1)) * pc3_scale - pc3_scale / 2
            z = np.random.random((num_pts, 1)) * pc3_scale - pc3_scale / 2
            xyz = np.hstack([x, y, z])
            shs = np.random.random((num_pts, 3)) / 255.0
            pcd_3 = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    elif use_hyper:
        cam_infos = readColmapCameras_hypernerf(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                                   images_folder=os.path.join(path, reading_dir))
        import json
        js1 = json.load(open(os.path.join(path, 'dataset.json'), 'r'))
        js2 = json.load(open(os.path.join(path, 'train_common.json'), 'r'))
        mono_cameras = js2['frame_names']

        if False:
            train_cam_infos= [this_cam for i, this_cam in enumerate(cam_infos) if(i % 4 == 0)]
            test_cam_infos= [this_cam for i, this_cam in enumerate(cam_infos) if((i+1) % 4 == 0)]
            val_cam_infos= [this_cam for i, this_cam in enumerate(cam_infos) if((i+2) % 4 + 2== 0)]
        else:
            # train_cam_infos= [this_cam for i, this_cam in enumerate(cam_infos) if this_cam.image_name in js1['train_ids']]
            train_cam_infos= [this_cam for i, this_cam in enumerate(cam_infos) if this_cam.image_name in mono_cameras]

            test_cam_infos= [this_cam for i, this_cam in enumerate(cam_infos) if this_cam.image_name in js1['val_ids']]
            val_cam_infos= [this_cam for i, this_cam in enumerate(cam_infos) if this_cam.image_name in js1['val_ids']]

        nerf_normalization = getNerfppNorm(train_cam_infos)

        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")

        using_colmap = True
        if using_colmap:

            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)

            storePly(ply_path, xyz, rgb)

            use_dense = False
            if use_dense:
                pc_in_aria = os.path.join(path, "fused.ply")
                import open3d as o3d
                pcd_aria = o3d.io.read_point_cloud(pc_in_aria)

                xyz = np.array(pcd_aria.points)



            shs = rgb / 255
            pcd = BasicPointCloud(points=xyz, colors=(shs), normals=np.zeros(np.shape(xyz)))
            pcd_max = xyz.max(axis=0) + 0.1 * nerf_normalization['radius']
            pcd_min = xyz.min(axis=0) - 0.1 * nerf_normalization['radius']
            num_pts = 30000
            print(f"Generating random point cloud ({num_pts})...")

            x = np.random.random((num_pts, 1)) * (pcd_max[0] - pcd_min[0]) + pcd_min[0]
            y = np.random.random((num_pts, 1)) * (pcd_max[1] - pcd_min[1]) + pcd_min[1]
            z = np.random.random((num_pts, 1)) * (pcd_max[2] - pcd_min[2]) + pcd_min[2]
            xyz_random = np.hstack([x, y, z])
            xyz = xyz_random
            shs = np.random.random((xyz.shape[0], 3))  # 255.0
            pcd_2 = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((len(xyz), 3)))
            pcd_3 = None


    scene_info = SceneInfo(point_cloud=pcd_2,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=[test_cam_infos[0]],
                           maxtime=0,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           point_cloud_second=pcd,
                           point_cloud_third=pcd_3,
                           val_cameras=val_cam_infos)

    return scene_info
def generateCamerasFromTransforms(path, template_transformsfile, extension, maxtime):
    trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

    rot_phi = lambda phi : torch.Tensor([
        [1,0,0,0],
        [0,np.cos(phi),-np.sin(phi),0],
        [0,np.sin(phi), np.cos(phi),0],
        [0,0,0,1]]).float()

    rot_theta = lambda th : torch.Tensor([
        [np.cos(th),0,-np.sin(th),0],
        [0,1,0,0],
        [np.sin(th),0, np.cos(th),0],
        [0,0,0,1]]).float()
    def pose_spherical(theta, phi, radius):
        c2w = trans_t(radius)
        c2w = rot_phi(phi/180.*np.pi) @ c2w
        c2w = rot_theta(theta/180.*np.pi) @ c2w
        c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
        return c2w
    cam_infos = []
    # generate render poses and times
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,160+1)[:-1]], 0)
    render_times = torch.linspace(0,maxtime,render_poses.shape[0])
    with open(os.path.join(path, template_transformsfile)) as json_file:
        template_json = json.load(json_file)
        try:
            fovx = template_json["camera_angle_x"]
        except:
            try:
                fovx = focal2fov(template_json['frames'][0]['fl_x'],template_json['frames'][0]['w'])
            except:
                fovx = focal2fov(template_json["fl_x"], template_json['w'])
    print("hello!!!!")
    # breakpoint()
    # load a single image to get image info.
    for idx, frame in enumerate(template_json["frames"]):
        cam_name = os.path.join(path, frame["file_path"] )
        image_path = os.path.join(path, cam_name)
        image_name = Path(cam_name).stem
        image = Image.open(image_path)
        im_data = np.array(image.convert("RGBA"))
        image = PILtoTorch(image,(800,800))
        break
    # format information
    for idx, (time, poses) in enumerate(zip(render_times,render_poses)):
        time = time/maxtime
        matrix = np.linalg.inv(np.array(poses))
        R = -np.transpose(matrix[:3,:3])
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]
        fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])
        FovY = fovy 
        FovX = fovx
        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=None, image_name=None, width=image.shape[1], height=image.shape[2],
                            time = time, mask=None))
    return cam_infos
def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", mapper = {}):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        try:
            fovx = contents["camera_angle_x"]
        except:
            try:

                fovx =focal2fov(contents['frames'][0]['fl_x'],contents['frames'][0]['w'])
            except:
                fovx = focal2fov(contents['fl_x'],contents['w'])

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"])
            try:
                time = mapper[frame["colmap_im_id"]]
            except:
                time = mapper[(frame['timestamp'] - frames[0]['timestamp']) * 1e-9]
            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3,:3])
            R[:,0] = -R[:,0]
            T = -matrix[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            # im_data = np.array(image.convert("RGBA"))
            #
            # bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
            #
            # norm_data = im_data / 255.0
            # arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            # image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            image = PILtoTorch(image,None)

            fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.shape[1], height=image.shape[2],
                            time = time, mask=None))
            
    return cam_infos
def read_timeline(path):
    with open(os.path.join(path, "transforms_train.json")) as json_file:
        train_json = json.load(json_file)
    with open(os.path.join(path, "transforms_test.json")) as json_file:
        test_json = json.load(json_file)
    try:
        time_line = [frame["colmap_im_id"] for frame in train_json["frames"]]
    except:
        time_line = [(frame['timestamp'] - train_json["frames"][0]['timestamp']) * 1e-9 for frame in train_json["frames"]]
    time_line = set(time_line)
    time_line = list(time_line)
    time_line.sort()
    timestamp_mapper = {}
    max_time_float = max(time_line)
    for index, time in enumerate(time_line):
        # timestamp_mapper[time] = index
        timestamp_mapper[time] = time/max_time_float

    return timestamp_mapper, max_time_float
def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    timestamp_mapper, max_time = read_timeline(path)
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, timestamp_mapper)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension, timestamp_mapper)
    print("Generating Video Transforms")
    video_cam_infos = generateCamerasFromTransforms(path, "transforms_train.json", extension, max_time)
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "fused.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 2000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    # storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
        pcd = fetchPly(ply_path)
        # xyz = -np.array(pcd.points)
        # pcd = pcd._replace(points=xyz)


    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=video_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=max_time
                           )
    return scene_info
def format_infos(dataset,split):
    # loading
    cameras = []
    image = dataset[0][0]
    if split == "train":
        for idx in tqdm(range(len(dataset))):
            image_path = None
            image_name = f"{idx}"
            time = dataset.image_times[idx]
            # matrix = np.linalg.inv(np.array(pose))
            R,T = dataset.load_pose(idx)
            FovX = focal2fov(dataset.focal[0], image.shape[1])
            FovY = focal2fov(dataset.focal[0], image.shape[2])
            cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                                time = time, mask=None,SD_feature=np.zeros(1)))

    return cameras


def readHyperDataInfos(datadir,use_bg_points,eval):
    train_cam_infos = Load_hyper_data(datadir,0.5,use_bg_points,split ="train")
    test_cam_infos = Load_hyper_data(datadir,0.5,use_bg_points,split="test")
    print("load finished")
    train_cam = format_hyper_data(train_cam_infos,"train")
    print("format finished")
    max_time = train_cam_infos.max_time
    video_cam_infos = copy.deepcopy(test_cam_infos)
    video_cam_infos.split="video"

    # ply_path = os.path.join(datadir, "points3D_downsample2.ply")

    ply_path = os.path.join(datadir, "fused.ply")
    pcd = fetchPly(ply_path)
    xyz = np.array(pcd.points)

    pcd = pcd._replace(points=xyz)
    nerf_normalization = getNerfppNorm(train_cam)
    plot_camera_orientations(train_cam_infos, pcd.points)
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=video_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=max_time
                           )

    return scene_info
def format_render_poses(poses,data_infos):
    cameras = []
    tensor_to_pil = transforms.ToPILImage()
    len_poses = len(poses)
    times = [i/len_poses for i in range(len_poses)]
    image = data_infos[0][0]
    for idx, p in tqdm(enumerate(poses)):
        # image = None
        image_path = None
        image_name = f"{idx}"
        time = times[idx]
        pose = np.eye(4)
        pose[:3,:] = p[:3,:]
        # matrix = np.linalg.inv(np.array(pose))
        R = pose[:3,:3]
        R = - R
        R[:,0] = -R[:,0]
        T = -pose[:3,3].dot(R)
        FovX = focal2fov(data_infos.focal[0], image.shape[2])
        FovY = focal2fov(data_infos.focal[0], image.shape[1])
        cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                            time = time, mask=None, SD_feature=np.zeros(1)))
    return cameras

def add_points(pointsclouds, xyz_min, xyz_max):
    add_points = (np.random.random((100000, 3)))* (xyz_max-xyz_min) + xyz_min
    add_points = add_points.astype(np.float32)
    addcolors = np.random.random((100000, 3)).astype(np.float32)
    addnormals = np.random.random((100000, 3)).astype(np.float32)
    # breakpoint()
    new_points = np.vstack([pointsclouds.points,add_points])
    new_colors = np.vstack([pointsclouds.colors,addcolors])
    new_normals = np.vstack([pointsclouds.normals,addnormals])
    pointsclouds=pointsclouds._replace(points=new_points)
    pointsclouds=pointsclouds._replace(colors=new_colors)
    pointsclouds=pointsclouds._replace(normals=new_normals)
    return pointsclouds
    # breakpoint()
    # new_
def readdynerfInfo(datadir,use_bg_points,eval):
    # loading all the data follow hexplane format
    # ply_path = os.path.join(datadir, "points3D_dense.ply")
    ply_path = os.path.join(datadir, "fused.ply")
    ply_ref = os.path.join(datadir, "fused_cleaned.ply")
    from scene.neural_3D_dataset_NDC import Neural3D_NDC_Dataset
    train_dataset = Neural3D_NDC_Dataset(
    datadir,
    "train",
    1.0,
    time_scale=1,
    scene_bbox_min=[-2.5, -2.0, -1.0],
    scene_bbox_max=[2.5, 2.0, 1.0],
    eval_index=0,
        )    
    test_dataset = Neural3D_NDC_Dataset(
    datadir,
    "test",
    1.0,
    time_scale=1,
    scene_bbox_min=[-2.5, -2.0, -1.0],
    scene_bbox_max=[2.5, 2.0, 1.0],
    eval_index=0,
        )


    train_cam_infos = format_infos(train_dataset,"train")
    val_cam_infos = format_render_poses(test_dataset.val_poses,test_dataset)
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # xyz = np.load

    pcd = fetchPly(ply_path)
    xyz = np.array(pcd.points)
    # xyz_len = list(range(xyz.shape[0]))
    # # choice_selected = np.random.choice(xyz_len, len(xyz_len) // 10, replace=False)
    # # xyz = xyz[choice_selected, :]
    # shs = np.random.random((xyz.shape[0], 3))  # / 255.0
    #
    # pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros(np.shape(xyz)))
    # pcd_max = [2.5, 2.0, 1.0]
    # pcd_min = [-2.5, -2.0, -1.0]

    # try:
    #     pcd_cleaned = fetchPly(ply_ref)
    #     xyz_cleaned = np.array(pcd_cleaned.points)
    #     pcd_max = pcd_cleaned.points.max(axis=0)
    #     pcd_min = pcd_cleaned.points.min(axis=0)
    #     num_pts = 400
    #     print(f"Generating random point cloud ({num_pts})...")
    #
    #     # We create random points inside the bounds of the synthetic Blender scees
    #     xyz_len = list(range(xyz_cleaned.shape[0]))
    #     choice_selected = np.random.choice(xyz_len, min(len(xyz_len) // 10, 100), replace=False)
    #     x = np.random.random((num_pts, 1)) * (pcd_max[0] - pcd_min[0]) + pcd_min[0]
    #     y = np.random.random((num_pts, 1)) * (pcd_max[1] - pcd_min[1]) + pcd_min[1]
    #     z = np.random.random((num_pts, 1)) * (pcd_max[2] - pcd_min[2]) + pcd_min[2]
    #     xyz_random = np.hstack([x, y, z])
    #
    #
    #     xyz = np.vstack((xyz_random,xyz_cleaned[choice_selected, :]))
    # except:
    if True:

        # all_cam_center = np.hstack( nerf_normalization['cam_centers']).T
        # all_cam_center.mean(axis=0)
        #
        # camera_max = all_cam_center.max(axis=0) + 5 * nerf_normalization['radius']
        # camera_min = all_cam_center.min(axis=0) - 5 * nerf_normalization['radius']
        #
        # pcd_max =  (np.vstack((camera_max, pcd.points.max(axis=0)))).min(axis=0)
        # pcd_min = (np.vstack((camera_min, pcd.points.min(axis=0)))).max(axis=0)

        all_cam_center = np.hstack( nerf_normalization['cam_centers']).T
        all_cam_center.mean(axis=0)

        camera_max = all_cam_center.max(axis=0) + 5 * nerf_normalization['radius']
        camera_min = all_cam_center.min(axis=0) - 5 * nerf_normalization['radius']

        pcd_max =  (np.vstack((camera_max, pcd.points.max(axis=0)))).min(axis=0)
        pcd_min = (np.vstack((camera_min, pcd.points.min(axis=0)))).max(axis=0)


        num_pts = 10000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz_len = list(range(xyz.shape[0]))
        choice_selected = np.random.choice(xyz_len, min(len(xyz_len) // 10, 100), replace=False)
        x = np.random.random((num_pts, 1)) * (pcd_max[0] - pcd_min[0]) + pcd_min[0]
        y = np.random.random((num_pts, 1)) * (pcd_max[1] - pcd_min[1]) + pcd_min[1]
        z = np.random.random((num_pts, 1)) * (pcd_max[2] - pcd_min[2]) + pcd_min[2]
        xyz_random = np.hstack([x, y, z])
        xyz = np.vstack((xyz[choice_selected, :], xyz_random))


    shs = np.random.random((xyz.shape[0], 3))  # 255.0
    pcd_2 = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((len(xyz), 3)))
    #
    # pcd_2 = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

    pc3_scale = 6

    x = np.random.random((num_pts, 1)) * pc3_scale - pc3_scale / 2
    y = np.random.random((num_pts, 1)) * pc3_scale - pc3_scale / 2
    z = np.random.random((num_pts, 1)) * pc3_scale - pc3_scale / 2
    xyz = np.hstack([x, y, z])
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd_3 = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))


    # scene_info = SceneInfo(point_cloud=pcd_2,
    #                        train_cameras=train_dataset,
    #                        test_cameras=test_dataset,
    #                        video_cameras=val_cam_infos,
    #                        nerf_normalization=nerf_normalization,
    #                        ply_path=ply_path,
    #                        point_cloud_second=pcd,
    #                        point_cloud_third=pcd_3,
    #                        val_cameras=test_dataset,
    #                        maxtime=300)
    scene_info = SceneInfo(point_cloud=pcd_2,
                           train_cameras=train_dataset,
                           test_cameras=test_dataset,
                           video_cameras=val_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=300,
                           point_cloud_second=pcd,
                           val_cameras=[test_dataset[0]]
                           )

    return  scene_info


def setup_camera(w, h, k, w2c, near=0.01, far=100):
    from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False,
        debug=True
    )
    return cam
def plot_camera_orientations(cam_list, xyz):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax2 = fig.add_subplot(122, projection='3d')
    # xyz = xyz[xyz[:,0]<1]
    threshold=2
    xyz = xyz[(xyz[:, 0] >= -threshold) & (xyz[:, 0] <= threshold) &
                         (xyz[:, 1] >= -threshold) & (xyz[:, 1] <= threshold) &
                         (xyz[:, 2] >= -threshold) & (xyz[:, 2] <= threshold)]

    ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c='r',s=0.1)
    for cam in tqdm(cam_list):
        # 提取 R 和 T
        R = cam.R
        T = cam.T

        direction = R @ np.array([0, 0, 1])

        ax.quiver(T[0], T[1], T[2], direction[0], direction[1], direction[2], length=1)

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.savefig("output.png")
    # breakpoint()
def readPanopticmeta(datadir, json_path):
    with open(os.path.join(datadir,json_path)) as f:
        test_meta = json.load(f)
    w = test_meta['w']
    h = test_meta['h']
    max_time = len(test_meta['fn'])
    cam_infos = []
    for index in range(len(test_meta['fn'])):
        focals = test_meta['k'][index]
        w2cs = test_meta['w2c'][index]
        fns = test_meta['fn'][index]
        cam_ids = test_meta['cam_id'][index]

        time = index / len(test_meta['fn'])
        # breakpoint()
        for focal, w2c, fn, cam in zip(focals, w2cs, fns, cam_ids):
            image_path = os.path.join(datadir,"ims")
            image_name=fn
            
            # breakpoint()
            image = Image.open(os.path.join(datadir,"ims",fn))
            im_data = np.array(image.convert("RGBA"))
            # breakpoint()
            im_data = PILtoTorch(im_data,None)[:3,:,:]
            # breakpoint()
            # print(w2c,focal,image_name)
            camera = setup_camera(w, h, focal, w2c)
            cam_infos.append({
                "camera":camera,
                "time":time,
                "image":im_data})
            
    cam_centers = np.linalg.inv(test_meta['w2c'][0])[:, :3, 3]  # Get scene radius
    scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))
    # breakpoint()
    return cam_infos, max_time, scene_radius 

def readPanopticSportsinfos(datadir):
    train_cam_infos, max_time, scene_radius = readPanopticmeta(datadir, "train_meta.json")
    test_cam_infos,_, _ = readPanopticmeta(datadir, "test_meta.json")
    nerf_normalization = {
        "radius":scene_radius,
        "translate":torch.tensor([0,0,0])
    }

    ply_path = os.path.join(datadir, "pointd3D.ply")

        # Since this data set has no colmap data, we start with random points
    plz_path = os.path.join(datadir, "init_pt_cld.npz")
    data = np.load(plz_path)["data"]
    xyz = data[:,:3]
    rgb = data[:,3:6]
    num_pts = xyz.shape[0]
    pcd = BasicPointCloud(points=xyz, colors=rgb, normals=np.ones((num_pts, 3)))
    storePly(ply_path, xyz, rgb)
    # pcd = fetchPly(ply_path)
    # breakpoint()
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=max_time,
                           )
    return scene_info

def readMultipleViewinfos(datadir,llffhold=8):

    cameras_extrinsic_file = os.path.join(datadir, "sparse_/images.bin")
    cameras_intrinsic_file = os.path.join(datadir, "sparse_/cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    from scene.multipleview_dataset import multipleview_dataset
    train_cam_infos = multipleview_dataset(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, cam_folder=datadir,split="train")
    test_cam_infos = multipleview_dataset(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, cam_folder=datadir,split="test")

    train_cam_infos_ = format_infos(train_cam_infos,"train")
    nerf_normalization = getNerfppNorm(train_cam_infos_)

    ply_path = os.path.join(datadir, "points3D_multipleview.ply")
    bin_path = os.path.join(datadir, "points3D_multipleview.bin")
    txt_path = os.path.join(datadir, "points3D_multipleview.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    
    try:
        pcd = fetchPly(ply_path)
        
    except:
        pcd = None
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=test_cam_infos.video_cam_infos,
                           maxtime=0,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "dynerf" : readdynerfInfo,
    "nerfies": readHyperDataInfos,  # NeRFies & HyperNeRF dataset proposed by [https://github.com/google/hypernerf/releases/tag/v0.1]
    "PanopticSports" : readPanopticSportsinfos,
    "MultipleView": readMultipleViewinfos
}
