_base_="default.py"
ModelParams=dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 100]

    },
)

OptimizationParams = dict(
    densify_grad_threshold_after=0.0004,
    densify_grad_threshold_coarse=0.0004,
    densify_grad_threshold_fine_init=0.0004,
    prune_small_foreground_visbility = True,
    downscale_mask_deform_lr = 0.1,
    accumulation_steps=1,
    eval_include_train_cams = True,
)
#### next reset SH or Not