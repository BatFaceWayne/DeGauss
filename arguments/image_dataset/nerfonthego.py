_base_="default.py"
ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
        'resolution': [64, 64, 64, 1000]
        # 'resolution': [180, 180, 180, 1440]
    },
)
OptimizationParams = dict(
    grid_lr_init= 0.0008,
    grid_lr_final=0.00002,
    densify_until_iter=26_000,
    lambda_depth_smoothness = 0.1,
    make_foreground_thresh_larger = True,
)