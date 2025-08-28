# DeGauss: Dynamic-Static Decomposition with Gaussian Splatting for Distractor-free 3D Reconstruction

## ICCV 2025

### [Project Page](https://batfacewayne.github.io/DeGauss.io/)| [arXiv Paper](https://arxiv.org/abs/2503.13176) | [Youtube](https://www.youtube.com/watch?v=d8U4--_jIcc) | [HuggingFace Model](https://huggingface.co/BatofGo/DeGauss_ckpts/tree/main)

[Rui Wang](https://pdz.ethz.ch/the-group/people/rui-wang.html) [Quentin Lohmeyer](https://pdz.ethz.ch/the-group/people/lohmeyer.html) [Mirko Meboldt](https://pdz.ethz.ch/the-group/people/meboldt.html) [Siyu Tang](https://vlg.ethz.ch/team/Prof-Dr-Siyu-Tang.html)

ETH Zurich


---

![Teaser](assets/teaser.jpg)

Our method achieves fast⚡️ and robust⛷️ dynamic-static decomposition based on 3D/4D gaussian splatting for a wide range of inputs as long egocentric videos, image collections, multi-view/monocular videos without extra supervision as optical flow.

---

![Mehtod](assets/method.png)
DeGauss simultaneously reconstructs the scene and learns an unsupervised decomposition into decoupled 3DGS background and 4DGS foreground branches based on their expressiveness. This design enables removing incorrectly modeled Gaussians in either branch during optimization, escaping local minima and generalizing to wide range of input data.

## News
2025.8.28: Release Aria preprocessing scripts

2025.7.31: Initial Code Release

## Environmental Setups

```bash
git clone git@github.com:BatFaceWayne/DeGauss.git
cd DeGauss
git submodule update --init --recursive
```

The environment fully compatitable with [4DGaussians](https://github.com/hustvl/4DGaussians). If you have that environment prepared you could skip this step. Alternatively you could prepare the environment as follows

```bash
conda create -n DeGauss python=3.11
conda activate DeGauss

pip install -r requirements.txt
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

## Data Preparation
Please follow the instructions in [4DGaussians](https://github.com/hustvl/4DGaussians) for Neu3D and HyperNerf dataset preparation, and [SpotLessSplats](https://github.com/lilygoli/SpotLessSplats/tree/main) for RobustNerf and Nerf on-the-go dataset processing. For Aria Datasets, please refer to [project-aria](https://www.projectaria.com/resources/#resources-datasets) and prepare data with [Nerfstudio](https://docs.nerf.studio/quickstart/custom_dataset.html#aria). The EPIC-Field dataset could be accessed [here](https://epic-kitchens.github.io/epic-fields/).


The dataset structureshould look follows

```
├── data
│   | hypernerf
│     ├── interp
│     ├── misc
│     ├── virg
│   | dynerf
│     ├── cook_spinach
│       ├── cam00
│           ├── images
│               ├── 0000.png
│               ├── 0001.png
│               ├── 0002.png
│               ├── ...
│     ...
|  | Nerf on-the-go/RobustNerf
|     ├── fountain
│       ├── images
│           ├── 1extra000.JPG
|           ...
│           ├── 2clutter008.JPG
|           ...
|     ├── mountain
|     ...
|  | Aria Digital Twin /Aria Everyday Activities / Hot3D
|     ├── Seq1
│       ├── images
│       ├── masks
│       |__ global_points.ply
|     ...

```
### Dataset processing
Note: The camera poses in [HyperNeRF](https://github.com/google/hypernerf) are rather inaccurate, as pointed out in [previous work](https://github.com/CVMI-Lab/SC-GS). Therefore we adopted colmap for pose estimation and only used this dataset for qualitative comparison. You could find the example of prepared Vrig-chicken scene [here](https://drive.google.com/file/d/1BoWvcSuQlGLdaO8iQIPuJFhiL1RldYhs/view?usp=drive_link).
<details>
  <summary>Aria dataset processing</summary>

Please refer to the [nerfstudio](https://docs.nerf.studio/quickstart/custom_dataset.html#aria) steps to prepare camera poses and extract fisheye frames.  Then create an "images_orig" folder and copy the raw fisheye images there. And copy the "transforms.json" as "transforms_orig.json".
The folder structure should look like this
```
├── Aria seq 1
│   ├── images_orig # original fisheye images
│   │   ├── camera-rgb_6469456023937.jpg
│   │   ├── camera-rgb_6469456023938.jpg
│   │   ├── ...
│   ├── transforms_orig.json #(fisheye camera style)
│   ├── global_points.ply

```
Then use the script in `scripts/linearize_aria.py` to transform the fisheye images into OPENCV camera format, and prepare COLMAP style input to be evaluated on various methods(including ours). 

```python
python scripts/linearize_aria.py --data path/to/aria/seq
```
After running this script you shoud ontain the structure as follows
```
├── Aria seq 1
│   ├── images_orig # original fisheye images
│   │   ├── camera-rgb_6469456023937.jpg
│   │   ├── camera-rgb_6469456023938.jpg
│   │   ├── ...
│   ├── transforms_orig.json #(fisheye camera style)
│   ├── global_points.ply
│   ├── images # undistorted images
│   │   ├── camera-rgb_6469456023937.jpg
│   │   ├── camera-rgb_6469456023938.jpg
│   │   ├── ...
│   ├── transforms.json #(OPENCV camera style)
│   ├── sparse ## COLMAP style OPENCV camera poses
│   │   ├── 0
│   │── masks  # Camera Masks, set up once at arguments/video_dataset/aria_data.py
│       ├── camera-rgb_6469456023937.png
│       ├── camera-rgb_6469456023938.png
│       ├── ...
```
</details>

## Checkpoints and Renders 🔥
To promote reproducibility, we have released our gaussian models, Full render and dynamic-static decomposition renders of Nerf on-the-go dataset, RobustNerf dataset and Neu3D dataset at [Checkpoints & Renders](https://huggingface.co/BatofGo/DeGauss_ckpts/tree/main). 

To visualize gaussian models, we recommend using this amazing gaussian splatting visualizing tool: [online visualizer](https://antimatter15.com/splat/).

## Training


For training video datasets as  `cut_roasted_beef` of Neu3D dataset, run
```python
##### please refer to the configs in folder arguments for different dataset setup
python train.py -s data/dynerf/cut_roasted_beef --port 6019 --expname cut_roasted_beef --configs arguments/video_dataset/Neu3D.py
```

For training image datasets for distractor-free scene modeling as Neu3D scenes such as `cut_roasted_beef`, run
```python
######## please use configs nerfonthego.py for indoor scenes
python3 train.py -s data/nerf-on-the-go/mountain --port 6019 --expname mountain --configs arguments/image_dataset/nerfonthego_outdoor.py
```
### Custom datasets
We have tested our method for various datasets, as reported in our paper. For customized dataset, simply prepare the input with colmap. And refer to the detailed hint in the arguments/video_dataset/default.py for parameters set-up.You can customize your training config through the config files.


## Rendering

Run the following script to render Neu3D dataset.
```python
######## please use configs nerfonthego.py for indoor scenes
python render_gaussian_dynerf.py -s path_to_dataset --port 6017 --expname Neu3Drender --configs
arguments/video_dataset/Neu3D.py" --render_checkpoint path_to_checkpoint
```


## Evaluation

You can just run the following script to evaluate the model.

```python
#### for dynamic scene eval -d : output base folder -s scene name
python calc_metric.py -d './test/' -s flame_steak_sparse

#### for distractor static scene eval -d : output base folder -s scene name
python calc_metric_static.py -d './test/' -s patio_high 

```


## Related Work

We sincerely thank the authors of following papers and their fantastic works

[4DGaussians:4D Gaussian Splatting for Real-Time Dynamic Scene Rendering](https://guanjunwu.github.io/4dgs/)

[SpotLessSplats: Ignoring Distractors in 3D Gaussian Splatting](https://spotlesssplats.github.io)


## Citation

If you find this repository/work helpful in your research, welcome to cite our paper and give a ⭐.

```
@misc{wang2025degaussdynamicstaticdecompositiongaussian,
        title={DeGauss: Dynamic-Static Decomposition with Gaussian Splatting for Distractor-free 3D Reconstruction}, 
        author={Rui Wang and Quentin Lohmeyer and Mirko Meboldt and Siyu Tang},
        year={2025},
        eprint={2503.13176},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/2503.13176}, 
  }
    
```
