#2023-07-26 02:35:43,285 - mmrotate - INFO - Epoch(val) [120][181]	mAP: 0.5426, AP50: 0.8860, AP55: 0.8060, AP60: 0.8030, AP65: 0.7930, AP70: 0.6920, AP75: 0.6460, AP80: 0.4370, AP85: 0.2490, AP90: 0.1120, AP95: 0.0010
#43.7
_base_ = [
    '../_base_/datasets/hrsc.py', '../_base_/schedules/schedule_6x.py',
    '../_base_/default_runtime.py'
]
angle_version = 'le90'

# model settings
model = dict(
    type='GroupKnowledgeDistillationRotatedFCOS',
    backbone=dict(
        type='ResNet',
        depth=34,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet34')),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='GroupKnowledgeDistillationRotatedFCOSHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        centerness_on_reg=True,
        separate_angle=False,
        scale_angle=True,
        group_num=3,
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', loss_weight=1.0),
        loss_angle=dict(type='L1Loss', loss_weight=0.06),
        loss_im=dict(type='IMLoss', loss_weight=0.3),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    group=dict(
        # group_cfg=[],
        # group_ckpt=[],
        is_teacher=False,
        group_cfg=['/media/dell/disk8/qaz/my_1/configs/rotated_fcos/rotated_fcos_r101_fpn_3x_hrsc_le90.py',
                   '/media/dell/disk8/qaz/my_1/configs/rotated_fcos/rotated_fcos_r50_fpn_1x_hrsc_le90.py',
                   '/media/dell/disk8/qaz/my_1/configs/rotated_fcos/rotated_fcos_r152_fpn_3x_hrsc_le90.py'],
        group_ckpt=['/media/dell/disk8/qaz/my_1/tools/work_dirs/rotated_fcos_r101_fpn_3x_hrsc_le90/latest.pth',
                    '/media/dell/disk8/qaz/my_1/tools/work_dirs/rotated_fcos_r50_fpn_1x_hrsc_le90/latest.pth',
                    '/media/dell/disk8/qaz/my_1/tools/work_dirs/rotated_fcos_r152_fpn_3x_hrsc_le90/latest.pth'],
    ),
    # training and testing settings
    train_cfg=None,
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000)
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
data = dict(
    train=dict(pipeline=train_pipeline, version=angle_version),
    val=dict(version=angle_version),
    test=dict(version=angle_version))