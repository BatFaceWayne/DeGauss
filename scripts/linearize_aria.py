#!/usr/bin/env python3
"""
Minimal cleanup of your original script:
- Swap WIDTH/HEIGHT when writing cameras.txt (COLMAP expects WIDTH, HEIGHT)
- Remove unused imports & duplicates
- Modernize array_to_blob / blob_to_array
- Fix integer division in pair_id_to_image_ids
- Add safe local import of mathutils inside get_position_details_json()

Everything else (flow/structure) is intentionally unchanged.
"""

import argparse
import json
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm


from nerfstudio.cameras.camera_utils import (
    fisheye624_project,
    fisheye624_unproject_helper,
)
from nerfstudio.data.utils.colmap_parsing_utils import (
    qvec2rotmat,
    rotmat2qvec,
    read_cameras_binary,
    read_images_binary,
    read_images_text,
    read_points3D_binary,
    read_points3D_text,
    write_cameras_binary,
    write_images_binary,
    write_points3D_binary,
    write_points3D_text,
)

import collections
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"]
)

# Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
# (License text omitted for brevity — identical to your original.)

MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = (
    f"""
CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {MAX_IMAGE_ID}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
"""
)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = (
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"
)

CREATE_ALL = "; ".join(
    [
        CREATE_CAMERAS_TABLE,
        CREATE_IMAGES_TABLE,
        CREATE_KEYPOINTS_TABLE,
        CREATE_DESCRIPTORS_TABLE,
        CREATE_MATCHES_TABLE,
        CREATE_TWO_VIEW_GEOMETRIES_TABLE,
        CREATE_NAME_INDEX,
    ]
)


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) // MAX_IMAGE_ID  # integer division
    return image_id1, image_id2


def array_to_blob(array: np.ndarray) -> bytes:
    return array.tobytes()


def blob_to_array(blob: bytes, dtype, shape=(-1,)) -> np.ndarray:
    return np.frombuffer(blob, dtype=dtype).reshape(*shape)


class COLMAPDatabase(sqlite3.Connection):
    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = lambda: self.executescript(
            CREATE_CAMERAS_TABLE
        )
        self.create_descriptors_table = lambda: self.executescript(
            CREATE_DESCRIPTORS_TABLE
        )
        self.create_images_table = lambda: self.executescript(
            CREATE_IMAGES_TABLE
        )
        self.create_two_view_geometries_table = lambda: self.executescript(
            CREATE_TWO_VIEW_GEOMETRIES_TABLE
        )
        self.create_keypoints_table = lambda: self.executescript(
            CREATE_KEYPOINTS_TABLE
        )
        self.create_matches_table = lambda: self.executescript(
            CREATE_MATCHES_TABLE
        )
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def add_camera(
        self,
        model,
        width,
        height,
        params,
        prior_focal_length=False,
        camera_id=None,
    ):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (
                camera_id,
                model,
                width,
                height,
                array_to_blob(params),
                int(prior_focal_length),
            ),
        )
        return cursor.lastrowid

    def add_image(
        self,
        name,
        camera_id,
        prior_q=np.full(4, np.NaN),
        prior_t=np.full(3, np.NaN),
        image_id=None,
    ):
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                image_id,
                name,
                camera_id,
                prior_q[0],
                prior_q[1],
                prior_q[2],
                prior_q[3],
                prior_t[0],
                prior_t[1],
                prior_t[2],
            ),
        )
        return cursor.lastrowid

    def add_keypoints(self, image_id, keypoints):
        assert len(keypoints.shape) == 2
        assert keypoints.shape[1] in [2, 4, 6]

        keypoints = np.asarray(keypoints, np.float32)
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints.shape + (array_to_blob(keypoints),),
        )

    def add_descriptors(self, image_id, descriptors):
        descriptors = np.ascontiguousarray(descriptors, np.uint8)
        self.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (image_id,) + descriptors.shape + (array_to_blob(descriptors),),
        )

    def add_matches(self, image_id1, image_id2, matches):
        assert len(matches.shape) == 2
        assert matches.shape[1] == 2

        if image_id1 > image_id2:
            matches = matches[:, ::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches),),
        )

    def add_two_view_geometry(
        self,
        image_id1,
        image_id2,
        matches,
        F=np.eye(3),
        E=np.eye(3),
        H=np.eye(3),
        qvec=np.array([1.0, 0.0, 0.0, 0.0]),
        tvec=np.zeros(3),
        config=2,
    ):
        assert len(matches.shape) == 2
        assert matches.shape[1] == 2

        if image_id1 > image_id2:
            matches = matches[:, ::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        qvec = np.asarray(qvec, dtype=np.float64)
        tvec = np.asarray(tvec, dtype=np.float64)
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,)
            + matches.shape
            + (
                array_to_blob(matches),
                config,
                array_to_blob(F),
                array_to_blob(E),
                array_to_blob(H),
                array_to_blob(qvec),
                array_to_blob(tvec),
            ),
        )


def example_usage():
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", default="database.db")
    args = parser.parse_args()

    if os.path.exists(args.database_path):
        print("ERROR: database path already exists -- will not modify it.")
        return

    db = COLMAPDatabase.connect(args.database_path)
    db.create_tables()

    model1, width1, height1, params1 = (
        0,
        1024,
        768,
        np.array((1024.0, 512.0, 384.0)),
    )
    model2, width2, height2, params2 = (
        2,
        1024,
        768,
        np.array((1024.0, 512.0, 384.0, 0.1)),
    )

    camera_id1 = db.add_camera(model1, width1, height1, params1)
    camera_id2 = db.add_camera(model2, width2, height2, params2)

    image_id1 = db.add_image("image1.png", camera_id1)
    image_id2 = db.add_image("image2.png", camera_id1)
    image_id3 = db.add_image("image3.png", camera_id2)
    image_id4 = db.add_image("image4.png", camera_id2)

    num_keypoints = 1000
    keypoints1 = np.random.rand(num_keypoints, 2) * (width1, height1)
    keypoints2 = np.random.rand(num_keypoints, 2) * (width1, height1)
    keypoints3 = np.random.rand(num_keypoints, 2) * (width2, height2)
    keypoints4 = np.random.rand(num_keypoints, 2) * (width2, height2)

    db.add_keypoints(image_id1, keypoints1)
    db.add_keypoints(image_id2, keypoints2)
    db.add_keypoints(image_id3, keypoints3)
    db.add_keypoints(image_id4, keypoints4)

    M = 50
    matches12 = np.random.randint(num_keypoints, size=(M, 2))
    matches23 = np.random.randint(num_keypoints, size=(M, 2))
    matches34 = np.random.randint(num_keypoints, size=(M, 2))

    db.add_matches(image_id1, image_id2, matches12)
    db.add_matches(image_id2, image_id3, matches23)
    db.add_matches(image_id3, image_id4, matches34)

    db.commit()

    rows = db.execute("SELECT * FROM cameras")
    camera_id, model, width, height, params, prior = next(rows)
    params = blob_to_array(params, np.float64)
    assert camera_id == camera_id1
    assert model == model1 and width == width1 and height == height1
    assert np.allclose(params, params1)

    camera_id, model, width, height, params, prior = next(rows)
    params = blob_to_array(params, np.float64)
    assert camera_id == camera_id2
    assert model == model2 and width == width2 and height == height2
    assert np.allclose(params, params2)

    keypoints = dict(
        (image_id, blob_to_array(data, np.float32, (-1, 2)))
        for image_id, data in db.execute("SELECT image_id, data FROM keypoints")
    )

    assert np.allclose(keypoints[image_id1], keypoints1)
    assert np.allclose(keypoints[image_id2], keypoints2)
    assert np.allclose(keypoints[image_id3], keypoints3)
    assert np.allclose(keypoints[image_id4], keypoints4)

    pair_ids = [
        image_ids_to_pair_id(*pair)
        for pair in (
            (image_id1, image_id2),
            (image_id2, image_id3),
            (image_id3, image_id4),
        )
    ]

    matches = dict(
        (pair_id_to_image_ids(pair_id), blob_to_array(data, np.uint32, (-1, 2)))
        for pair_id, data in db.execute("SELECT pair_id, data FROM matches")
    )

    assert np.all(matches[(image_id1, image_id2)] == matches12)
    assert np.all(matches[(image_id2, image_id3)] == matches23)
    assert np.all(matches[(image_id3, image_id4)] == matches34)

    db.close()

    if os.path.exists(args.database_path):
        os.remove(args.database_path)


def createdbfromjson():
    # Placeholder kept to match your original structure.
    db = COLMAPDatabase.connect('/home/ray/Downloads/aria_kitchen_simpler_test/database.db')
    db.create_tables()
    db.close()


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def get_position_details_json(object):
    import mathutils  # local import to avoid global dependency if unused

    world_matrix = object.matrix_world
    blender_to_colmap_rotation = np.diag([1, -1, -1])
    blender_world_translation, blender_world_rotation, blender_world_scale = world_matrix.decompose()
    blender_view_rotation = blender_world_rotation.to_matrix().transposed()
    blender_view_translation = -1.0 * blender_view_rotation @ blender_world_translation
    colmap_view_rotation = blender_to_colmap_rotation @ blender_view_rotation
    colmap_view_rotation_quaternion = mathutils.Matrix(colmap_view_rotation).to_quaternion()
    colmap_view_translation = blender_to_colmap_rotation @ blender_view_translation
    return {
        "name": object.name,
        "x_pos": colmap_view_translation[0],
        "y_pos": colmap_view_translation[1],
        "z_pos": colmap_view_translation[2],
        "w_rotation": colmap_view_rotation_quaternion.w,
        "x_rotation": colmap_view_rotation_quaternion.x,
        "y_rotation": colmap_view_rotation_quaternion.y,
        "z_rotation": colmap_view_rotation_quaternion.z,
    }


def write_camera_poses_to_text_file(file_path, camera_details):
    try:
        with open(file_path, 'w') as write_file:
            camera_details.sort(key=lambda x: x['name'].strip('progcam'))
            for i, camera in enumerate(camera_details):
                write_file.writelines(
                    f"{i+1} {camera['w_rotation']} {camera['x_rotation']} {camera['y_rotation']} {camera['z_rotation']} {camera['x_pos']} {camera['y_pos']} {camera['z_pos']} 1 {camera['name']}.png\n\n"
                )
    except Exception as e:
        print(e)


# --- Optional utility kept from your code; may rely on external helpers not shown here ---

def colmap_to_json(
    recon_dir: Path,
    output_dir: Path,
    camera_mask_path: Optional[Path] = None,
    image_id_to_depth_path: Optional[Dict[int, Path]] = None,
    image_rename_map: Optional[Dict[str, str]] = None,
    ply_filename: str = "sparse_pc.ply",
    keep_original_world_coordinate: bool = False,
) -> int:
    cam_id_to_camera = read_cameras_binary(recon_dir / "cameras.bin")
    im_id_to_image = read_images_binary(recon_dir / "images.bin")

    frames = []
    for im_id, im_data in im_id_to_image.items():
        rotation = qvec2rotmat(im_data.qvec)
        translation = im_data.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(w2c)
        c2w[0:3, 1:3] *= -1
        if not keep_original_world_coordinate:
            c2w = c2w[np.array([0, 2, 1, 3]), :]
            c2w[2, :] *= -1

        name = im_data.name
        if image_rename_map is not None:
            name = image_rename_map[name]
        name = Path(f"./images/{name}")

        frame = {
            "file_path": name.as_posix(),
            "transform_matrix": c2w.tolist(),
            "colmap_im_id": im_id,
        }
        if camera_mask_path is not None:
            frame["mask_path"] = camera_mask_path.relative_to(camera_mask_path.parent.parent).as_posix()
        if image_id_to_depth_path is not None:
            depth_path = image_id_to_depth_path[im_id]
            frame["depth_file_path"] = str(depth_path.relative_to(depth_path.parent.parent))
        frames.append(frame)

    if set(cam_id_to_camera.keys()) != {1}:
        raise RuntimeError("Only single camera shared for all images is supported.")
    # NOTE: parse_colmap_camera_params() and create_ply_from_colmap() are not defined in this snippet; kept as-is.
    out = parse_colmap_camera_params(cam_id_to_camera[1])  # noqa: F821
    out["frames"] = frames

    applied_transform = None
    if not keep_original_world_coordinate:
        applied_transform = np.eye(4)[:3, :]
        applied_transform = applied_transform[np.array([0, 2, 1]), :]
        applied_transform[2, :] *= -1
        out["applied_transform"] = applied_transform.tolist()

    assert ply_filename.endswith(".ply"), f"ply_filename: {ply_filename} does not end with '.ply'"
    create_ply_from_colmap(  # noqa: F821
        ply_filename,
        recon_dir,
        output_dir,
        torch.from_numpy(applied_transform).float() if applied_transform is not None else None,
    )
    out["ply_file_path"] = ply_filename

    with open(output_dir / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)

    return len(frames)


# --- local text parsers (kept) ---

def read_cameras_text(path):
    cameras = {}
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            camera_id = int(elems[0])
            model = elems[1]
            width = int(elems[2])
            height = int(elems[3])
            params = np.array(tuple(map(float, elems[4:])))
            cameras[camera_id] = Camera(id=camera_id, model=model, width=width, height=height, params=params)
    return cameras


def forward_trans(qvec, tvec):
    rotation = qvec2rotmat(qvec)
    translation = tvec.reshape(3, 1)
    w2c = np.concatenate([rotation, translation], 1)
    w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
    c2w = np.linalg.inv(w2c)
    c2w[0:3, 1:3] *= -1
    c2w = c2w[np.array([0, 2, 1, 3]), :]
    c2w[2, :] *= -1
    return c2w


def backward_trans(c2w):
    c2w = c2w.copy()
    c2w[2, :] *= -1
    c2w = c2w[np.array([0, 2, 1, 3]), :]
    c2w[0:3, 1:3] *= -1
    w2c = np.linalg.inv(c2w)
    return w2c


def _undistort_image(
    fisheye_crop_radius, distortion_params, frame, image, K
):
    mask = None

    # Build fisheye624 params
    fisheye624_params = torch.from_numpy(np.array(
        [frame['fl_x'], frame['fl_y'], frame['cx'], frame['cy']] + distortion_params
    )).float()
    assert fisheye624_params.shape == (16,)

    upper, lower, left, right = fisheye624_unproject_helper(
        torch.tensor(
            [
                [frame['cx'], frame['cy'] - fisheye_crop_radius],
                [frame['cx'], frame['cy'] + fisheye_crop_radius],
                [frame['cx'] - fisheye_crop_radius, frame['cy']],
                [frame['cx'] + fisheye_crop_radius, frame['cy']],
            ],
            dtype=torch.float32,
        )[None],
        params=fisheye624_params[None],
    ).squeeze(dim=0)
    fov_radians = torch.max(
        torch.acos(torch.sum(upper * lower / torch.linalg.norm(upper) / torch.linalg.norm(lower))),
        torch.acos(torch.sum(left * right / torch.linalg.norm(left) / torch.linalg.norm(right))),
    )

    undist_h = int(fisheye_crop_radius * 2)
    undist_w = int(fisheye_crop_radius * 2)
    undistort_focal = undist_h / (2 * torch.tan(fov_radians / 2.0))
    undist_K = torch.eye(3)
    undist_K[0, 0] = undistort_focal
    undist_K[1, 1] = undistort_focal
    undist_K[0, 2] = (undist_w - 1) / 2.0
    undist_K[1, 2] = (undist_h - 1) / 2.0

    undist_uv_homog = torch.stack(
        [
            *torch.meshgrid(
                torch.arange(undist_w, dtype=torch.float32),
                torch.arange(undist_h, dtype=torch.float32),
            ),
            torch.ones((undist_w, undist_h), dtype=torch.float32),
        ],
        dim=-1,
    )
    assert undist_uv_homog.shape == (undist_w, undist_h, 3)
    dist_uv = (
        fisheye624_project(
            xyz=(
                torch.einsum(
                    "ij,bj->bi",
                    torch.linalg.inv(undist_K),
                    undist_uv_homog.reshape((undist_w * undist_h, 3)),
                )[None]
            ),
            params=fisheye624_params[None, :],
        )
        .reshape((undist_w, undist_h, 2))
        .numpy()
    )
    map1 = dist_uv[..., 1]
    map2 = dist_uv[..., 0]

    image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)

    dist_h = frame['h']
    dist_w = frame['w']
    mask = np.mgrid[:dist_h, :dist_w]
    mask[0, ...] -= dist_h // 2
    mask[1, ...] -= dist_w // 2
    mask = np.linalg.norm(mask, axis=0) < fisheye_crop_radius
    mask = torch.from_numpy(
        cv2.remap(
            mask.astype(np.uint8) * 255,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        / 255.0
    ).bool()[..., None]
    if len(mask.shape) == 2:
        mask = mask[:, :, None]
    assert mask.shape == (undist_h, undist_w, 1)
    K = undist_K.numpy()

    return K, image, mask


# --------------------
# Main (kept as-is)
# --------------------

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="database.db")
args = parser.parse_args()
base_dir = args.data
in_json = os.path.join(base_dir, 'transforms_orig.json')
output_dir = os.path.join(base_dir, 'manual_sparse')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(base_dir, 'sparse_gen'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'masks'), exist_ok=True)
nerfstudio_json = json.load(open(in_json, "r"))
nerfstudio_json_data = nerfstudio_json.copy()
camera_details = []
ariatonerf = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
write_file = open(os.path.join(output_dir, 'images.txt'), 'w')

name_frame_unsorted = [float(temp_f['file_path'].split('/')[-1].split('_')[-1].split('.')[0]) for temp_f in nerfstudio_json['frames']]
idx_sorted = np.argsort(name_frame_unsorted)

for temp_i in range(len(idx_sorted)):
    frame = nerfstudio_json['frames'][idx_sorted[temp_i]]
    i = idx_sorted[temp_i]
    c2w = np.array(frame['transform_matrix'])
    c2w[2, :] *= -1
    c2w = c2w[np.array([0, 2, 1, 3]), :]
    c2w[0:3, 1:3] *= -1
    w2c = np.linalg.inv(c2w)
    rotation = w2c[0:3, 0:3]
    translation = w2c[0:3, 3]
    qvec = rotmat2qvec(rotation)
    tvec = translation
    write_file.writelines(
        f"{i + 1} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {translation[0]} {translation[1]} {translation[2]} 1 {frame['file_path'].split('/')[-1]}\n\n"
    )
write_file.close()

with open(os.path.join(output_dir, 'cameras.txt'), "w") as fid:
    print('undistorting images')
    frames_all = nerfstudio_json['frames']

    for i, frame in enumerate(tqdm(
                frames_all,
                total=len(frames_all),
                desc='Undistorting the aria fisheye images',
                unit='img',
                dynamic_ncols=True,
        )):
        K = np.eye(3)
        K[0, 0] = frame['fl_x']
        K[1, 1] = frame['fl_y']
        K[0, 2] = frame['cx']
        K[1, 2] = frame['cy']
        distortion_params = frame['distortion_params']
        this_imageout = os.path.join(base_dir, 'images', frame['file_path'].split('/')[-1])
        if not os.path.exists(this_imageout) or i < 1:
            image = cv2.imread(os.path.join(base_dir, 'images_orig', frame['file_path'].split('/')[-1]))[:, :, ::-1]
            K, image, mask = _undistort_image(nerfstudio_json['fisheye_crop_radius'], distortion_params, frame, image, K)
            cv2.imwrite(os.path.join(base_dir, 'images', frame['file_path'].split('/')[-1]), image[:, :, ::-1])
            if mask is not None:
                cv2.imwrite(os.path.join(base_dir, 'masks', frame['file_path'].split('/')[-1]).split('.')[0] + '.png', mask.numpy() * 255)

        # NOTE: WIDTH=image.shape[1], HEIGHT=image.shape[0]
        to_write = [
            str(1), 'OPENCV',
            str(image.shape[1]), str(image.shape[0]),
            str(K[0, 0]), str(K[1, 1]), str(K[0, 2]), str(K[1, 2])
        ] + [0, 0, 0, 0]
        line = " ".join([str(elem) for elem in to_write])
        fid.write(line + "\n")

test_read = read_cameras_text(os.path.join(output_dir, 'cameras.txt'))
test_images = read_images_text(os.path.join(output_dir, 'images.txt'))
write_cameras_binary(test_read, os.path.join(output_dir, 'cameras.bin'))
key_list = np.array(list(range(1, len(test_images.keys()) + 1)))
new_test_images = {}

db = COLMAPDatabase.connect(os.path.join(base_dir, 'database.db'))
db.create_tables()
nerfstudio_json = json.load(open(in_json, "r"))
# Uses last K/image from loop (kept as in your original)
model1, width1, height1, params1 = (
    4,
    image.shape[1],
    image.shape[0],
    np.array(([K[0, 0], K[1, 1], K[0, 2], K[1, 2]] + [0, 0, 0, 0])),
)
camera_id1 = db.add_camera(model1, width1, height1, params1)

for temp_i in range(len(idx_sorted)):
    frame = nerfstudio_json['frames'][idx_sorted[temp_i]]
    i = idx_sorted[temp_i]
    db.add_image(frame['file_path'].split('/')[-1], camera_id1)

db.commit()
db.close()

# ----- generate new transformation json file (kept) -----
new_list = []
new_data = nerfstudio_json_data.copy()
new_data['camera_model'] = 'OPENCV'
try:
    del new_data['fisheye_crop_radius']
except Exception:
    pass

for idx_temp in range(len(nerfstudio_json_data['frames'])):
    i = idx_sorted[idx_temp]
    NEW_path = os.path.join(base_dir, 'images', nerfstudio_json_data['frames'][i]['file_path'].split('/')[-1])
    if os.path.exists(NEW_path):
        tempf = nerfstudio_json_data['frames'][i].copy()
        tempf['file_path'] = NEW_path
        tempf['fl_x'] = float(K[0, 0])
        tempf['fl_y'] = float(K[1, 1])
        tempf['cx'] = float(K[0, 2])
        tempf['cy'] = float(K[1, 2])
        tempf['w'] = float(image.shape[1])
        tempf['h'] = float(image.shape[0])
        del tempf['distortion_params']
        new_list.append(tempf)

new_data['frames'] = new_list
print(len(new_list))
json_out = os.path.join(base_dir, 'transforms.json')
with open(json_out, 'w') as f:
    json.dump(new_data, f)

# points3D.txt stub
open(os.path.join(output_dir, 'points3D.txt'), 'w').close()

# write bin files
write_cameras_binary(read_cameras_text(os.path.join(output_dir, 'cameras.txt')), os.path.join(output_dir, 'cameras.bin'))
write_images_binary(read_images_text(os.path.join(output_dir, 'images.txt')), os.path.join(output_dir, 'images.bin'))

# copy to final colmap layout
final_colmap_dir = os.path.join(base_dir, 'sparse', '0')
os.makedirs(final_colmap_dir, exist_ok=True)
shutil.copyfile(os.path.join(base_dir, 'manual_sparse', 'cameras.bin'), os.path.join(final_colmap_dir, 'cameras.bin'))
shutil.copyfile(os.path.join(base_dir, 'manual_sparse', 'images.bin'), os.path.join(final_colmap_dir, 'images.bin'))

# Optional COLMAP steps kept disabled
#### use this if you want to force COLMAP to re-triangulate the points and get a colored Point Cloud
# if False:
#     os.system('colmap feature_extractor --database_path ' + os.path.join(base_dir, 'database.db') + ' --image_path ' + os.path.join(base_dir, 'images'))
#     os.system("colmap sequential_matcher --database_path " + os.path.join(base_dir, 'database.db'))
#     os.system("colmap point_triangulator --database_path " + os.path.join(base_dir, 'database.db') + " --image_path " + os.path.join(base_dir, 'images')
#               + " --input_path " + os.path.join(base_dir, 'manual_sparse')
#               + " --output_path " + os.path.join(base_dir, 'sparse_gen'))
#     shutil.copyfile(os.path.join(base_dir, 'sparse_gen', 'points3D.bin'), os.path.join(final_colmap_dir, 'points3D.bin'))
