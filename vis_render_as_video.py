import os
import cv2
import numpy as np
import imageio
from matplotlib import cm
from tqdm import tqdm


def process_images(seq_name, base_path='./test'):
    """
    Process mask and light images, compute results, and save visualization images.
    """
    in_dir = os.path.join(base_path, seq_name)
    dirs = {
        'gt': os.path.join(in_dir, 'gt'),
        'mask': os.path.join(in_dir, 'mask_comp'),
        'pred': os.path.join(in_dir, 'full_pred'),
        'static': os.path.join(in_dir, 'static_light'),
        'dy_vis': os.path.join(in_dir, 'image_dyvis'),
        'mask_vis': os.path.join(in_dir, 'mask_vis'),
        'static_dispose': os.path.join(in_dir, 'static_dispose')
    }

    # Ensure output directories exist
    for key in ['dy_vis', 'mask_vis', 'static_dispose']:
        os.makedirs(dirs[key], exist_ok=True)

    for file_name in os.listdir(dirs['gt']):
        if file_name.startswith('.'):
            continue

        ref_name = os.path.splitext(file_name)[0]

        # Input paths
        mask_path = os.path.join(dirs['mask'], ref_name + '.npy')
        pred_path = os.path.join(dirs['pred'], ref_name + '.png')
        static_path = os.path.join(dirs['static'], ref_name + '.png')

        # Output paths
        result_path = os.path.join(dirs['dy_vis'], ref_name + '.png')
        mask_vis_path = os.path.join(dirs['mask_vis'], ref_name + '.png')
        static_dispose_path = os.path.join(dirs['static_dispose'], ref_name + '.png')

        # Load inputs
        mask = np.load(mask_path)
        mask_copy = mask.copy()
        full_pred = imageio.imread(pred_path)
        static_light = imageio.imread(static_path)

        # Ensure mask is 3D for broadcasting
        if mask.ndim == 2 and full_pred.ndim == 3:
            mask = mask[..., np.newaxis]

        # Compute outputs
        result = full_pred - (static_light * (1 - mask))
        static_dispose = static_light * (1 - mask)

        # Clip and save
        result = np.clip(result, 0, 255).astype(np.uint8)
        static_dispose = np.clip(static_dispose, 0, 255).astype(np.uint8)
        imageio.imwrite(result_path, result)
        imageio.imwrite(static_dispose_path, static_dispose)

        # Visualize mask
        jet_colormap = cm.get_cmap('jet')
        color_image = jet_colormap(mask_copy[:, :, 0])[:, :, :3]
        color_image = (np.clip(color_image * 255, 0, 255)).astype(np.uint8)
        imageio.imwrite(mask_vis_path, color_image)


def create_video(seq_name, base_path='./test', fps=30):
    """
    Create a video by horizontally concatenating corresponding frames from specified folders.
    """
    parent_folder = os.path.join(base_path, seq_name)
    output_video = os.path.join(parent_folder, 'render.mp4')

    subfolder_order = ["gt", "full_pred", "image_dyvis", "static_raw", "mask_vis"]

    # Validate folders
    for subfolder in subfolder_order:
        full_path = os.path.join(parent_folder, subfolder)
        if not os.path.isdir(full_path):
            raise FileNotFoundError(f"Missing subfolder: {subfolder}")

    # Load image paths
    image_lists = []
    for subfolder in subfolder_order:
        folder_path = os.path.join(parent_folder, subfolder)
        images = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and 'left' not in f
        ])
        if not images:
            raise ValueError(f"No images found in subfolder: {subfolder}")
        image_paths = [os.path.join(folder_path, f) for f in images]
        image_lists.append(image_paths)

    # Check consistency
    num_frames = len(image_lists[0])
    for idx, img_list in enumerate(image_lists):
        if len(img_list) != num_frames:
            raise ValueError(f"Mismatch in frame count for: {subfolder_order[idx]}")

    # Write video
    writer = imageio.get_writer(output_video, fps=fps)
    for i in tqdm(range(num_frames), desc="Creating video"):
        frames = []
        for img_path in [lst[i] for lst in image_lists]:
            img = cv2.imread(img_path)
            if img is None:
                raise IOError(f"Failed to read image: {img_path}")
            frames.append(img)
        combined = np.hstack(frames)
        rgb_frame = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        writer.append_data(rgb_frame)

    writer.close()
    print(f"✅ Video saved at: {output_video}")


if __name__ == "__main__":
    ###### after training the sequence, visualize the dynamic-static decomposition as a video
    SEQ_NAME = 'cut_roasted_beefcc'
    BASE_PATH = './test'

    process_images(SEQ_NAME, BASE_PATH)
    create_video(SEQ_NAME, BASE_PATH)
