# model settings
import copy
import os
import liauto.projects.end2end as end2end_cfg

DEBUG = bool(int(os.environ.get("DEBUG", False)))
EVAL_WITH_PNP = bool(int(os.environ.get("EVAL_WITH_PNP", False)))
ONLY_INFER_PNP = bool(int(os.environ.get("ONLY_INFER_PNP", False)))
PARALLEL = bool(int(os.environ.get("PARALLEL", False)))
FREEZE_COMMON = bool(int(os.environ.get("FREEZE_COMMON", False)))

FREEZE_BACKBONE = bool(int(os.environ.get("FREEZE_BACKBONE", False))) or FREEZE_COMMON
FREEZE_NECK = bool(int(os.environ.get("FREEZE_NECK", False))) or FREEZE_COMMON
FREEZE_VIEW_TRANS = (
    bool(int(os.environ.get("FREEZE_VIEW_TRANS", False))) or FREEZE_COMMON
)
FREEZE_PRE_BEV = bool(int(os.environ.get("FREEZE_PRE_BEV", False))) or FREEZE_COMMON
FREEZE_REDUCE_BEV = bool(int(os.environ.get("FREEZE_REDUCE_BEV", False)))
FREEZE_OCC = bool(int(os.environ.get("FREEZE_OCC", False)))
FREEZE_CONE = bool(int(os.environ.get("FREEZE_CONE", False)))
FREEZE_DYNAMIC = bool(int(os.environ.get("FREEZE_DYNAMIC", False)))
FREEZE_STATIC = bool(int(os.environ.get("FREEZE_STATIC", False)))
FREEZE_DET2D = bool(int(os.environ.get("FREEZE_DET2D", False)))
FREEZE_PTS = bool(int(os.environ.get("FREEZE_PTS", False))) or FREEZE_COMMON
FREEZE_PNP = bool(int(os.environ.get("FREEZE_PNP", False)))
NO_GRAD_NORM = bool(int(os.environ.get("NO_GRAD_NORM", False)))
AUTO_EVAL = bool(int(os.environ.get("AUTO_EVAL", False)))
VOXEL_DETERMINISTIC = bool(int(os.environ.get("VOXEL_DETERMINISTIC", True)))
FINETUNE = bool(int(os.environ.get("FINETUNE", False)))
LPAI_PIPELINE = bool(int(os.environ.get("LPAI_PIPELINE", False)))


cfg_root = os.path.dirname(end2end_cfg.__file__)

# only for static
file_client_args = dict(backend="disk")

############## IMG Config ###############
data_config = {
    "cams": [
        "CAM_FRONT_LEFT",
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT",
        "CAM_BACK",
        "CAM_BACK_RIGHT",
        "CAM_FRONT_CROP",
    ],
    "use_yuv": False,
    "Ncams": 7,
    "input_size": (512, 960),
    "src_size": (2160, 3840),
    "crop_input_size": (2160, 3840),
    "crop_size": (512, 960),
    "crop_shift_range": (80, 150),  # (h_shift, w_shift)
    # Augmentation
    "resize": (-0.02, 0.11),  # (-0.06, 0.11)
    "rot": (-1.6, 1.6),  # (-5.4, 5.4)
    "flip": False,  # True
    "crop_h": (0.0, 0.0),
    "resize_test": 0.00,
    "do_hue_aug": True,
    "do_hue_ratio": 0.3,
    "brightness_delta": 16,
    "contrast_range": (0.8, 1.2),
    "saturation_range": (0.8, 1.2),
    "hue_delta": 9,
    "sharpen": (-0.2, 0.5),
    "drop_ratio": 0.01,
    "random_resample_prob": 0.3,  # random interpolation probability
}


############## BEV Config ###############
# ****** OneModel BEV *******
grid_config = {
    "x": [-79.8, 102.6, 0.6],
    "y": [-50.4, 50.4, 0.6],
    "z": [-3.5, 4.5, 8],
    "depth": [1.0, 115.0, 0.75],
}
voxel_size = [0.075, 0.075, 0.2]
grid_size = [
    int((grid_config["x"][1] - grid_config["x"][0]) / voxel_size[0]),
    int((grid_config["y"][1] - grid_config["y"][0]) / voxel_size[1]),
    int((grid_config["z"][1] - grid_config["z"][0]) / voxel_size[2]),
]
out_size_factor = grid_config["x"][2] / voxel_size[0]


def decompose_grid_configs(grid_config, voxel_size, regression_max_grid=20):
    out_size_factor = (
        grid_config["x"][2] / voxel_size[0],
        grid_config["y"][2] / voxel_size[1],
    )
    assert out_size_factor[0] == out_size_factor[1]

    pc_range = [
        grid_config["x"][0],
        grid_config["y"][0],
        grid_config["z"][0],
        grid_config["x"][1],
        grid_config["y"][1],
        grid_config["z"][1],
    ]
    post_center_range = [
        pc_range[0] - voxel_size[0] * regression_max_grid,
        pc_range[1] - voxel_size[1] * regression_max_grid,
        pc_range[2],
        pc_range[3] + voxel_size[0] * regression_max_grid,
        pc_range[4] + voxel_size[1] * regression_max_grid,
        pc_range[5],
    ]
    return out_size_factor[0], pc_range, post_center_range


# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
# point_cloud_range = [0, -24.0, -1, 79.2, 24.0, 2.6]
# point_cloud_range = [-66.80, -61.44, -3.5, 86.80, 61.44, 4.5]
# point_cloud_range = [-79.20, -52.8, -3.5, 84.00, 52.8, 4.5]
point_cloud_range = [-79.8, -50.4, -3.5, 102.6, 50.4, 4.5]
post_center_range = [-89.8, -60.4, -10, 150, 60.4, 10]
use_BDA = True
bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5), scale_lim=(0.95, 1.05), flip_dx_ratio=0.5, flip_dy_ratio=0.5
)
skip_frame_num = 5
multi_adj_frame_id_cfg = (1, 22 + 1, skip_frame_num)

############## Common Model Settings ###############
numC_Trans = 96
se_fusion = True
with_motion_states = True
focus_special_object_weight = 2
with_light_states = True
norm_cfg = dict(type="BN", requires_grad=True)
norm_config_mmt = dict(type="BN", requires_grad=True, momentum=0.01)
input_modality = dict(
    use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=False
)
obj_class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

############## Backbone ###############
backbone_out_inds = [0, 1, 2, 3]
channels_list = [64, 128, 256, 512]
img_backbone = dict(
    # pretrained=os.environ.get("PRETRAIN", "torchvision://resnet34"),
    type="ResNet",
    depth=34,
    num_stages=4,
    out_indices=backbone_out_inds,
    frozen_stages=-1,
    norm_cfg=norm_config_mmt,
    norm_eval=False,
    with_cp=False,
    style="pytorch",
    __graph_model_name="backbone",
    __freeze=FREEZE_BACKBONE,
    __freeze_bn=FREEZE_BACKBONE,
)

############## Necks ###############
det_spliter = dict(
    type="NeckSpliter",
    input_idxs=[1, 2, 3],
    __graph_model_name="det_spliter",
    __freeze=False,
)
det_2d_spliter = dict(
    type="NeckSpliter",
    input_idxs=[0],
    __graph_model_name="det_2d_spliter",
    __freeze=False,
)
seg_spliter = dict(
    type="NeckSpliter",
    input_idxs=[0, 1, 2],
    __graph_model_name="seg_spliter",
    __freeze=False,
)
bev_spliter = dict(
    type="NeckSpliter",
    input_idxs=[0],
    __graph_model_name="bev_spliter",
    __freeze=False,
)
bev_spliter_dynamic = dict(
    type="NeckSpliter",
    input_idxs=[0, 1, 2],
    __graph_model_name="bev_spliter_dynamic",
    __freeze=False,
)
bev_spliter_static = dict(
    type="NeckSpliter",
    input_idxs=[0],
    __graph_model_name="bev_spliter_static",
    __freeze=False,
)

start_idx = 1
end_idx = 4
neck_in_channels = [channels_list[i] for i in backbone_out_inds[start_idx:end_idx]]
neck_out_channels = 256
neck_extra_conv_channels = 256
img_neck = dict(
    type="CustomYOLOXPAFPN",
    in_channels=neck_in_channels,
    # start_idx=start_idx,
    # end_idx=end_idx,
    norm_cfg=norm_config_mmt,
    out_channels=neck_out_channels,
    num_csp_blocks=1,
    __graph_model_name="det_neck",
    __freeze=FREEZE_NECK,
    __freeze_bn=FREEZE_NECK,
)

seg_neck_inds_of_backbone_outs = [0, 1, 2]
seg_neck_inds = [backbone_out_inds[i] for i in seg_neck_inds_of_backbone_outs]
seg_neck_in_channels = [channels_list[i] for i in seg_neck_inds]

neck_ratio = 0.25
arm_out_chs = [int(x * neck_ratio) for x in [64, 96, 128]]
cm_out_ch = int(128 * neck_ratio)
seg_neck = dict(
    type="PPLiteSegNeck",
    in_indices=seg_neck_inds_of_backbone_outs,
    in_channels=seg_neck_in_channels,
    arm_type="UAFM_SpAtten",
    arm_out_chs=arm_out_chs,
    cm_bin_sizes=[1, 2, 4],
    cm_out_ch=cm_out_ch,
    resize_mode="nearest",
    __graph_model_name="seg_neck",
    __freeze=False,
)

############## Det_Head ###############
det_head_in_channels = neck_extra_conv_channels
det_head = dict(
    type="ClippedYOLOXHead",
    num_classes=1,
    in_channels=det_head_in_channels,
    feat_channels=64,
    norm_cfg=norm_cfg,
    strides=[8],  # (8, 16, 32),
    train_cfg=dict(assigner=dict(type="SimOTAAssigner", center_radius=2.5)),
    test_cfg=dict(score_thr=0.01, nms=dict(type="nms", iou_threshold=0.65)),
    __graph_model_name="det_head",
    __freeze=FREEZE_DET2D,
    __freeze_bn=FREEZE_DET2D,
)

############## Seg_Head ###############
seg_head_in_channels = arm_out_chs[0]
seg_head = dict(
    type="PPLiteSegSimpleHead",
    in_channels=seg_head_in_channels,
    in_index=0,
    channels=64,
    dropout_ratio=0,
    num_classes=4,
    norm_cfg=norm_cfg,
    align_corners=False,
    sampler=dict(type="OHEMPixelSampler", thresh=0.7, min_kept=100000),
    loss_decode=[
        dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
        dict(type="LovaszLoss", per_image=False, loss_weight=8.0, reduction="none"),
    ],
)

aux_chs = arm_out_chs[1:]
seg_auxiliary_head = dict(
    type="MultiAuxSegHead",
    head_configs=[
        {**seg_head, "in_index": i + 1, "in_channels": ch}
        for i, ch in enumerate(aux_chs)
    ],
    __graph_model_name="aux_head",
    __freeze=False,
)

img_view_in_channels = neck_extra_conv_channels
############## ImgViewTransformer ###############
img_view_transformer = dict(
    type="LSSViewTransformerBEVDepth_V2",
    grid_config=grid_config,
    input_size=data_config["input_size"],
    in_channels=img_view_in_channels,
    out_channels=numC_Trans,
    depthnet_cfg=dict(use_dcn=False, mid_cut_ratio=8),
    downsample=8,
    use_fisheye_undistort=True,
    loss_depth_weight=10 / grid_config["depth"][1],
    normalize_mlp=True,
    __graph_model_name="bev_img_view_transformer",
    __freeze=FREEZE_VIEW_TRANS,
    __freeze_bn=FREEZE_VIEW_TRANS,
)

############## PreProcess ###############
pre_process = dict(
    type="CustomResNet",
    numC_input=numC_Trans,
    num_layer=[1],
    num_channels=[numC_Trans if se_fusion else 256],
    norm_cfg=norm_config_mmt,
    stride=[1],
    backbone_output_ids=[0],
    numC_trans=numC_Trans,
    __graph_model_name="bev_pre_process",
    __freeze=FREEZE_PRE_BEV,
    __freeze_bn=FREEZE_PRE_BEV,
)

reduce_bev = dict(
    type="ReduceBEV",
    numC_trans=numC_Trans,
    norm_cfg=norm_config_mmt,
    __graph_model_name="bev_reduce_bev",
    __freeze=FREEZE_REDUCE_BEV,
    __freeze_bn=FREEZE_REDUCE_BEV,
)

############## ImgViewTransformerStatic ###############
grid_config_static = {
    "x": [-19.8, 119.4, 0.6],
    "y": [-24.0, 24.0, 0.6],
    "z": [-3.5, 4.5, 8.0],  # unused param
    "depth": [1.0, 129.0, 1.0],
}
numC_Trans_static = 128
img_view_transformer_static = dict(
    type="LSSViewTransformerBEVDepth_V2",
    grid_config=grid_config_static,
    input_size=data_config["input_size"],
    in_channels=img_view_in_channels,
    out_channels=numC_Trans_static,
    depthnet_cfg=dict(use_dcn=False, mid_cut_ratio=256 / 128),
    downsample=8,
    use_fisheye_undistort=True,
    loss_depth_weight=10 / grid_config_static["depth"][1],
    normalize_mlp=True,
    __graph_model_name="bev_img_view_transformer_static",
    __freeze=FREEZE_STATIC,
    __freeze_bn=FREEZE_STATIC,
)
############## PreProcessStatic ###############
pre_process_static = dict(
    type="CustomResNet",
    numC_input=numC_Trans_static,
    num_layer=[1],
    num_channels=[numC_Trans_static if se_fusion else 256],
    norm_cfg=norm_config_mmt,
    stride=[1],
    backbone_output_ids=[0],
    numC_trans=numC_Trans_static,
    __graph_model_name="bev_pre_process_static",
    __freeze=FREEZE_STATIC,
    __freeze_bn=FREEZE_STATIC,
)
reduce_bev_static = dict(
    type="ConvGRU",
    input_size=numC_Trans_static,
    hidden_sizes=[numC_Trans_static],
    kernel_sizes=[3],
    n_layers=1,
    __graph_model_name="bev_reduce_bev_static",
    __freeze=FREEZE_STATIC,
    __freeze_bn=FREEZE_STATIC,
)

############## BEVBackbone ###############
img_bev_encoder_backbone = dict(
    type="CustomResNet",
    # numC_input=numC_Trans * (len(range(*multi_adj_frame_id_cfg))+1),
    numC_input=numC_Trans if se_fusion else 256 * 2,
    num_channels=[numC_Trans * 2, numC_Trans * 3],
    num_layer=[2, 2],
    stride=[2, 2],
    norm_cfg=norm_config_mmt,
    numC_trans=numC_Trans,
    __graph_model_name="bev_img_bev_encoder_backbone",
    __freeze=FREEZE_DYNAMIC,
    __freeze_bn=FREEZE_DYNAMIC,
)

############## BEVBackbone ###############
img_bev_encoder_backbone_occ = dict(
    type="CustomResNet",
    # numC_input=numC_Trans * (len(range(*multi_adj_frame_id_cfg))+1),
    numC_input=numC_Trans if se_fusion else 256 * 2,
    num_channels=[numC_Trans * 2, numC_Trans * 4],
    num_layer=[2, 2],
    stride=[2, 2],
    norm_cfg=norm_config_mmt,
    numC_trans=numC_Trans,
    __graph_model_name="bev_img_bev_encoder_backbone_occ",
    __freeze=FREEZE_OCC,
    __freeze_bn=FREEZE_OCC,
)

############## StaticBEVBackbone ###############
img_bev_encoder_backbone_static = dict(
    type="CustomResNet",
    # numC_input=numC_Trans * (len(range(*multi_adj_frame_id_cfg))+1),
    numC_input=numC_Trans_static if se_fusion else 256 * 2,
    num_channels=[numC_Trans_static * 2, numC_Trans_static * 4, numC_Trans_static * 8],
    norm_cfg=norm_config_mmt,
    numC_trans=numC_Trans_static,
    __graph_model_name="bev_img_bev_encoder_backbone_static",
    __freeze=FREEZE_STATIC,
    __freeze_bn=FREEZE_STATIC,
)

############## MLCBEVBackbone ###############
img_bev_encoder_backbone_static_mlc = dict(
    type="CustomResNet",
    # numC_input=numC_Trans * (len(range(*multi_adj_frame_id_cfg))+1),
    numC_input=numC_Trans_static if se_fusion else 256 * 2,
    num_channels=[numC_Trans_static * 2, numC_Trans_static * 4, numC_Trans_static * 8],
    norm_cfg=norm_config_mmt,
    se_fusion=se_fusion,
    numC_trans=numC_Trans_static,
    __graph_model_name="bev_img_bev_encoder_backbone_static_mlc",
    __freeze=FREEZE_STATIC,
    __freeze_bn=FREEZE_STATIC,
)

############## BEVNeck ###############
out_channels_bev_neck_dynamic = 128
img_bev_encoder_neck = dict(
    type="FPN_LSS",
    in_channels=numC_Trans * 3 + numC_Trans * 2,
    out_channels=out_channels_bev_neck_dynamic,
    input_feature_index=(0, 1),
    scale_factor=2,
    norm_cfg=norm_config_mmt,
    __graph_model_name="bev_img_bev_encoder_neck",
    __freeze=FREEZE_DYNAMIC,
    __freeze_bn=FREEZE_DYNAMIC,
)

############## OccBEVNeck ###############
img_bev_encoder_neck_occ = dict(
    type="FPN_LSS",
    in_channels=numC_Trans * 4 + numC_Trans * 2,
    out_channels=128,
    input_feature_index=(0, 1),
    scale_factor=2,
    norm_cfg=norm_config_mmt,
    __graph_model_name="bev_img_bev_encoder_neck_occ",
    __freeze=FREEZE_OCC,
    __freeze_bn=FREEZE_OCC,
)

############## StaticBEVNeck ###############
img_bev_encoder_neck_static = dict(
    type="FPN_LSS",
    in_channels=numC_Trans_static * 8 + numC_Trans_static * 2,
    norm_cfg=norm_config_mmt,
    out_channels=128,
    __graph_model_name="bev_img_bev_encoder_neck_static",
    __freeze=FREEZE_STATIC,
    __freeze_bn=FREEZE_STATIC,
)

############## MLCBEVNeck ###############
img_bev_encoder_neck_static_mlc = dict(
    type="FPN_LSS",
    in_channels=numC_Trans_static * 8 + numC_Trans_static * 2,
    norm_cfg=norm_config_mmt,
    out_channels=256,
    __graph_model_name="bev_img_bev_encoder_neck_static_mlc",
    __freeze=FREEZE_STATIC,
    __freeze_bn=FREEZE_STATIC,
)

############## Fusion ###############
at128_point_cloud_range = [1.8, -50.4, -3.5, 150.0, 50.4, 4.5]
at128_voxel_size = [0.3, 0.3, 8]
pts_bvh_shape = [
    int(
        round(
            (at128_point_cloud_range[4] - at128_point_cloud_range[1])
            / at128_voxel_size[1]
        )
    ),  # y
    int(
        round(
            (at128_point_cloud_range[3] - at128_point_cloud_range[0])
            / at128_voxel_size[0]
        )
    ),  # x
]
voxel_feat_numC = 32

pts_voxel_encoder = dict(
    type="LiHardVFE",
    in_channels=4,
    feat_channels=[voxel_feat_numC],
    with_distance=False,
    voxel_size=at128_voxel_size,
    with_cluster_center=False,
    with_voxel_center=True,
    point_cloud_range=at128_point_cloud_range,
    point_cloud_norm=[150, 50, 5, 255, 1, 1, 4],  # xyzi center_xyz
    # norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01),
    __graph_model_name="pts_voxel_encoder",
    __freeze=FREEZE_PTS,
    __freeze_bn=FREEZE_PTS,
    __pts_model=True,
)

pts_middle_encoder = dict(
    type="PointPillarsScatter",
    in_channels=voxel_feat_numC,
    output_shape=pts_bvh_shape,
    __graph_model_name="pts_middle_encoder",
    __freeze=FREEZE_PTS,
    __freeze_bn=FREEZE_PTS,
    __pts_model=True,
)

pts_backbone = dict(
    type="SECOND",
    in_channels=voxel_feat_numC,
    # norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
    layer_nums=[3, 5, 5],
    layer_strides=[2, 2, 2],
    out_channels=[64, 128, 256],
    __graph_model_name="pts_backbone",
    __freeze=FREEZE_PTS,
    __freeze_bn=FREEZE_PTS,
    __pts_model=True,
)

static_voxel_size = [0.6, 0.6, 8]
pts_target_shape = (
    int(
        round(
            (at128_point_cloud_range[4] - at128_point_cloud_range[1])
            / static_voxel_size[1]
        )
    ),  # y
    int(
        round(
            (at128_point_cloud_range[3] - at128_point_cloud_range[0])
            / static_voxel_size[0]
        )
    ),  # x
)
pts_neck = dict(
    type="PTS_FPN_SKIP",
    in_channels=voxel_feat_numC * 2,
    out_channels=numC_Trans,
    norm_cfg=norm_config_mmt,
    extra_upsample=1,
    __graph_model_name="pts_neck",
    __freeze=FREEZE_PTS,
    __freeze_bn=FREEZE_PTS,
)

pts_neck_upsample = dict(
    type="PTS_NECK_UPSAMPLE",
    extra_upsample=2,
    __graph_model_name="pts_neck_upsample",
    __freeze=FREEZE_PTS,
    __freeze_bn=FREEZE_PTS,
)

pts_neck_upsample_static = dict(
    type="PTS_NECK_UPSAMPLE_STAIC",
    target_size=pts_target_shape,
    __graph_model_name="pts_neck_upsample_static",
    __freeze=False,
    __freeze_bn=False,
)

pts_neck_convblock = dict(
    type="PTS_FPN_SKIP_CONVBLOCK",
    in_channels=64,
    out_channels=numC_Trans,
    norm_cfg=norm_config_mmt,
    __graph_model_name="pts_neck_convblock",
    __freeze=FREEZE_PTS,
    __freeze_bn=FREEZE_PTS,
)

pts_neck_convblock_static = dict(
    type="PTS_FPN_SKIP_CONVBLOCK",
    in_channels=64,
    out_channels=numC_Trans_static,
    norm_cfg=norm_config_mmt,
    __graph_model_name="pts_neck_convblock_static",
    __freeze=FREEZE_STATIC,
    __freeze_bn=FREEZE_STATIC,
)

pts_segmentation_occ = dict(
    type="SE_ASPPSegHead",
    inplanes=128,
    planes=128,
    nbr_classes=1,
    dilations_conv_list=[1, 6, 12, 18],
    threshold=0.5,
    pts_seg_loss_cfg=dict(
        downsample_ratio=2,
        noise_index=8,
        noise_thresh=0.01,
        gt_crop_range=[58, 110, 0, 129],
        pred_crop_range=[27, 156],
        loss_cfgs=[
            dict(
                type="CrossEntropyLoss",
                use_sigmoid=True,
                class_weight=2,
                loss_weight=1.0,
            ),
            dict(
                type="LovaszLoss",
                loss_type="binary",
                reduction="none",
                per_image=False,
                loss_weight=1.0,
            ),
            dict(
                type="DiceLoss",
                loss_weight=0.2,
            ),
            dict(type="GaussianFocalLoss", alpha=2.5, gamma=3.5, reduction="mean"),
            dict(
                type="PtsL1Loss",
                loss_weight=0.1,
            ),
        ],
    ),
    __graph_model_name="pts_segmentation_occ",
    __freeze=FREEZE_OCC,
    __freeze_bn=FREEZE_OCC,
)

############## Datasets ###############
cone_det_dataset = dict(
    type="Front30JoinWeightedDataset",
    anno_list=None,
    dir_root=os.environ.get("LPAI_INPUT_DATA_0"),
    classes_map=None,
    ignore_heavily_occlude=False,
    ignore_invisible=True,
    # box_threshold=20,
    transform_configs=[
        dict(
            type="ComposeTransform",
            target_source="front30",
            trans_list=[
                [
                    dict(
                        type="Resize",
                        img_scale=(1920, 1080),
                        keep_ratio=True,
                        ratio_range=[0.8, 1.2],
                    ),
                    dict(
                        type="RandomCenterCropPad",
                        ratios=[1.0],
                        crop_size=(512, 960),
                        mean=(0, 0, 0),
                        std=(1, 1, 1),
                        to_rgb=False,
                        test_pad_mode=None,
                        center_shift_bound=[40, 40],
                        # border=[256-20, 480-40]
                    ),
                ],
                [
                    dict(
                        type="Resize",
                        img_scale=(960, 540),
                        keep_ratio=True,
                        ratio_range=[0.8, 1.2],
                    ),
                    dict(
                        type="RandomCenterCropPad",
                        ratios=[1.0],
                        crop_size=(512, 960),
                        mean=(0, 0, 0),
                        std=(1, 1, 1),
                        to_rgb=False,
                        test_pad_mode=None,
                        center_shift_bound=[10, 10],
                        # border=[256-20, 480-40]
                    ),
                ],
            ],
        ),
        dict(
            type="ComposeTransform",
            target_source="front120",
            trans_list=[
                [
                    dict(
                        type="Resize",
                        img_scale=(1920, 1080),
                        keep_ratio=True,
                        ratio_range=[0.8, 1.2],
                    ),
                    dict(
                        type="RandomCenterCropPad",
                        ratios=[1.0],
                        crop_size=(512, 960),
                        mean=(0, 0, 0),
                        std=(1, 1, 1),
                        to_rgb=False,
                        test_pad_mode=None,
                        center_shift_bound=[40, 40],
                        # border=[256-20, 480-40]
                    ),
                ],
                [
                    dict(
                        type="Resize",
                        img_scale=(960, 540),
                        keep_ratio=True,
                        ratio_range=[0.8, 1.2],
                    ),
                    dict(
                        type="RandomCenterCropPad",
                        ratios=[1.0],
                        crop_size=(512, 960),
                        mean=(0, 0, 0),
                        std=(1, 1, 1),
                        to_rgb=False,
                        test_pad_mode=None,
                        center_shift_bound=[10, 10],
                        # border=[256-20, 480-40]
                    ),
                ],
            ],
        ),
        dict(
            type="RandomFlip",
            flip_ratio=0.5,
            direction="horizontal",
        ),
        dict(
            type="FilterBboxByClsAndMergeCls",
            cls_filter=[1, 2, -2],
        ),
        dict(
            type="FilterBboxByHeight",
            min_height=10,
        ),
        # dict(type="Visualization2D"),
        dict(
            type="Normalize",
            mean=[128.0, 128.0, 128.0],
            std=[128.0, 128.0, 128.0],
            to_rgb=True,
        ),
        dict(type="DefaultFormatBundle"),
        dict(
            type="Collect",
            keys=["img", "gt_bboxes", "gt_labels", "gt_bboxes_ignore", "img_metas"],
            meta_keys=["ori_shape", "img_shape", "scale", "scale_factor"],
        ),
    ],
    min_base_image_count=10000,
)

det_dataset_resize = dict(
    type="DetectionWeightedDataset",
    anno_list=None,
    dir_root=os.environ.get("LPAI_INPUT_DATA_0"),
    classes_map=None,
    transform_configs=[
        dict(
            type="Resize", img_scale=(960, 540), keep_ratio=True, ratio_range=[0.8, 1.0]
        ),
        dict(
            type="RandomCenterCropPad",
            ratios=[1.0],
            crop_size=(512, 960),
            mean=(0, 0, 0),
            std=(1, 1, 1),
            to_rgb=False,
            test_pad_mode=None,
            center_shift_bound=[40, 40],
            # border=[256-20, 480-40]
        ),
        dict(
            type="RandomFlip",
            flip_ratio=0.3,
            direction="horizontal",
        ),
        dict(
            type="Normalize",
            mean=[128.0, 128.0, 128.0],
            std=[128.0, 128.0, 128.0],
            to_rgb=True,
        ),
        dict(type="FilterBboxByHeight", min_height=10),
        dict(type="DefaultFormatBundle"),
        dict(
            type="Collect",
            keys=["img", "gt_bboxes", "gt_labels", "gt_bboxes_ignore"],
            meta_keys=["ori_shape", "img_shape", "scale", "scale_factor", "keep_ratio"],
        ),
    ],
    min_base_image_count=50000,
)
det_dataset_crop = copy.deepcopy(det_dataset_resize)
det_dataset_crop["transform_configs"][0] = dict(
    type="Resize", img_scale=(3840, 2160), keep_ratio=True, ratio_range=[0.8, 1.0]
)

seg_dataset_resize = dict(
    type="SegmentationWeightedDataset",
    anno_list=None,
    dir_root=os.environ.get("LPAI_INPUT_DATA_0"),
    num_classes=None,
    classes_map=None,
    ignore_index=255,
    transform_configs=[
        dict(
            type="Resize",
            img_scale=(960, 540),
            keep_ratio=False,
            ratio_range=[1.0, 1.0],
        ),
        dict(
            type="RandomCrop",
            crop_size=(512, 960),
            crop_type="absolute",
        ),
        dict(
            type="RandomFlip",
            flip_ratio=0.3,
            direction="horizontal",
        ),
        dict(
            type="Normalize",
            mean=[128.0, 128.0, 128.0],
            std=[128.0, 128.0, 128.0],
            to_rgb=True,
        ),
        dict(type="SegDefaultFormatBundle"),
        dict(
            type="Collect",
            keys=["img", "gt_semantic_seg"],
            meta_keys=["ori_shape", "img_shape", "scale", "scale_factor", "keep_ratio"],
        ),
    ],
    min_base_image_count=50000,
)
seg_dataset_crop = copy.deepcopy(seg_dataset_resize)
seg_dataset_crop["transform_configs"][0] = dict(
    type="Resize", img_scale=(3840, 2160), keep_ratio=False, ratio_range=[1.0, 1.0]
)

det_infer_cfg = dict(
    name=None,
    task="detection",
    class_name=[],
    reg_type="yolox",
    branches=["cls_score", "pred_bbox", "objectness"],
    stride=[8],  # [8, 16, 32],
    score_threshold=[0.5],
    element_type=5,  # TODO hard code
    progress=0,  # TODO hard code
    output_batch_idx=None,
)
