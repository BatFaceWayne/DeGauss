_base_="default.py"
ModelParams=dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 150]

    },
)

OptimizationParams = dict(

    coarse_iterations = 2000,
    coarse_mean_override =True,

    densify_grad_threshold_after=0.0004,
    densify_grad_threshold_coarse=0.0004,
    densify_grad_threshold_fine_init=0.0004,

    ###### reduce model complexity by pruning dynamic gaussians with small foreground visibility
    prune_small_foreground_visbility = True,
    downscale_mask_deform_lr = 0.01,
    separation_high_prob = True,
    separation_low_prob = True,
    lambda_main_loss_= True
)
#### next reset SH or Not