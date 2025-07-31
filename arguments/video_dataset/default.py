
ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
    'resolution': [64, 64, 64, 150] ##### ideally the forth entry to be half the sequence length eg: 150 for 300 frames
    },

    # less grid multi res more efficient normally [1, 2] ; [1, 2, 4] works fine
    multires = [1,2,4],
    defor_depth = 1,
    net_width = 128,
    plane_tv_weight = 0.0002,
    time_smoothness_weight = 0.001,
    l1_time_planes =  0.0001,
    render_process=False,
    no_do=False,
    no_dshs=False,
    no_ds=False,
)
OptimizationParams = dict(
    ##### saving folder for training process visualziation and final
    ########################################################
    ####### important arguments, these arguments are important for reconstruction quality
    ########################################################

    ###### coarse stage is important to regularize deformation prediction and separate dynamic from statics
    ###### max(1000, seq len/2)
    coarse_iterations=1000,
    ######## this is coarse mean override forces composition probability = 0.5 in coarse stage, Disable would promote foreground init
    coarse_mean_override = False,

    ######### grid_lr should be larger to promote dynamic learning, but too large lr would lead to all one/ all zero prediction.
    ######## reduce lr as needed, normally  0.00016 init and 0.000016 final would work.
    ##### grid learning rate init
    grid_lr_init=0.0008,
    ##### grid learning rate final
    grid_lr_final=0.000005,

    ######### using motion mask grad for densification, useful with sparse image inputs
    use_motion_grad=False,
    ##### SH learning rate downscaling start, /20 default. set to 2 to regularize foreground modeling.
    ###### However using 20 also works for video. But recommend set to 2 for images
    SH_lr_downscaling_start=8,
    SH_lr_downscaling_end=20,
    ###### Normally larger is better
    batch_size=2,
    ###### Useful especially for image inputs
    accumulation_steps = 1,
    ##### encourage dynamic-static decomposition prediction normally 4 is ok
    lambda_main_loss_=4,
    ############################################################

    saving_folder='./test/',
    ##### max iterations
    iterations = 20_000,

    separation_high_prob = True,
    densify_until_iter = 16_000,
    ##### opacity reset interval, change if needed. More frequent reset for cleaner
    ##### scene but reduced quality
    opacity_reset_interval = 2000,
    ##### position learning rate max steps
    position_lr_max_steps = 20_000,
    ##### feature learning rate default
    feature_lr=0.0025,

    opacity_threshold_coarse = 0.005,
    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.005,

    ####### reset SH for more explicit pruning
    reset_SH = False,

    pruning_interval = 600,
    eval_include_train_cams = False,


)