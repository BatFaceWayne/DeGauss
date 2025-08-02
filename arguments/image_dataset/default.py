ModelHiddenParams = dict(
    kplanes_config={
        'grid_dimensions': 2,
        'input_coordinate_dim': 4,
        'output_coordinate_dim': 16,
        'resolution': [64, 64, 64, 150]
    },
    # less grid multi res more efficient
    multires=[1, 2],
    defor_depth=1,
    net_width=128,
    plane_tv_weight=0.0002,
    time_smoothness_weight=0.001,
    l1_time_planes=0.0001,
    render_process=False,
    no_do=False,
    no_dshs=False,
    no_ds=False,
)
OptimizationParams = dict(
    ##### saving folder for training process visualziation and final
    saving_folder='./test/',
    ##### max iterations
    iterations=30_000,
    ##### batch size for training
    batch_size=4,
    ##### coarse iterations for training
    coarse_iterations=1000,
    ##### feature learning rate
    ##### densify until iter
    densify_until_iter=24_000,
    ##### opacity reset interval, change if needed. More frequent reset for cleaner
    ##### scene but reduced quality
    opacity_reset_interval=1000,
    ##### position learning rate max steps
    position_lr_max_steps=20_000,
    ##### feature learning rate default
    feature_lr=0.0025,
    ##### grid learning rate final
    grid_lr_final=0.00002,
    ##### opacity threshold default
    opacity_threshold_coarse=0.005,
    opacity_threshold_fine_init=0.005,
    opacity_threshold_fine_after=0.005,
    ##### foreground oneupshinterval
    foreground_oneupshinterval=200,
    ##### background oneupshinterval, slower background gaussian SH to explicitly suppress floaters modeling
    background_oneupshinterval=2000,
    ##### use penal large gaussians
    use_penal_large_gaussians=False,
    ##### use penal spiky gaussians
    use_penal_spiky_gaussians=False,
    ##### SH learning rate downscaling start, /20 default. set to 2 to regularize foreground modeling
    SH_lr_downscaling_start=2,
    SH_lr_downscaling_end=2,
    ###### accumulation steps for training for sparse image collections
    accumulation_steps=4,
    lambda_main_loss=4,
    ##### use motion grad for image collections
    use_motion_grad=True,
    use_depth_smoothness_loss=True,

    ###### encourage foreground modeling where with large structural difference than background
    penalize_dynamic=False,
    ####### reset SH for more explicit pruning
    reset_SH=False,
    ####### weight for penal light end, encourge brightness control to be 1, allowing
    ####### light modeling with SH coefficients without inducing floaters in early stages
    weight_penal_light_end=0.2,
    ####### encourge foreground prob to be either 0 or 1
    lambda_entropy_loss=0.001,
    ####### max gaussian foreground, change if needed
    max_gaussian_foreground=160000,
    ####### max gaussian background, change if needed
    max_gaussian_background=2660000,
    pruning_interval=100
)