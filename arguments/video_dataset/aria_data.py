_base_="default.py"
ModelParams=dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [256, 256, 256, 1400]

    },
    multires=[1, 2, 4],
    defor_depth=1,
    net_width=128,
    no_do=False,
    no_dshs=False,
    no_ds=False,
)

OptimizationParams = dict(
    iterations = 120_000,
    coarse_iterations = 1000,
    densify_until_iter = 100_000,
    position_lr_max_steps = 120_000,
    #### we suggest grid lr and deformation lr to be larger for better separation for long monocular video
    #### but consider lower this if the training is unstable
    grid_lr_init= 0.0016,
    grid_lr_final=0.00016,
    deformation_lr_init=0.00016,
    deformation_lr_final=0.000016,
    densify_grad_threshold_after=0.0002,
    densify_grad_threshold_coarse=0.0002,
    densify_grad_threshold_fine_init=0.0002,
    prune_small_foreground_visbility = False,
    downscale_mask_deform_lr = 0.1,
    accumulation_steps=1,
    reset_SH = False,
    lambda_main_loss_=4,
    SH_lr_downscaling_start=8,
    SH_lr_downscaling_end=8,
    lambda_depth_smoothness=0.1,
    opacity_reset_interval = 4000,
    pruning_interval = 300,
    weight_penal_light_end=0.1,
    separation_high_prob = True,
    detach_background_separation = False,
    eval_include_train_cams = True,
    vignette_mask = './assets/vignette_imx577.png',
    camera_mask = './assets/aria_camera_mask.png',
    foreground_oneupshinterval = 1000,
    background_oneupshinterval = 10000,
    lambda_loss_depth_back = 1,
    max_gaussian_foreground = 100000,
    max_gaussian_background = 360000,
    make_background_thresh_larger = False


)
