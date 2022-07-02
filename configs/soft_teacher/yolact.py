checkpoint_config = dict(interval=4000, by_epoch=False, max_keep_ckpts=20)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='pre_release',
                name='soft_teacher_yolact_r101_1x8_coco',
                config=dict(
                    percent=95,
                    work_dirs='work_dirs/${cfg_name}/${percent}',
                    total_step=180000)),
            by_epoch=False)
    ])
custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='WeightSummary'),
    dict(type='MeanTeacher', momentum=0.999, interval=1, warm_up=0)
]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
img_size = 550
model = dict(
    type='SoftTeacher',
    model=dict(
        type='YOLACT',
        backbone=dict(
            type='ResNet',
            depth=101,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=-1,
            norm_cfg=dict(type='BN', requires_grad=False),
            norm_eval=True,
            zero_init_residual=False,
            style='caffe',
            init_cfg=dict(
                type='Pretrained',
                checkpoint='open-mmlab://detectron2/resnet101_caffe')),
        neck=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=1,
            add_extra_convs='on_input',
            num_outs=5,
            upsample_cfg=dict(mode='bilinear')),
        bbox_head=dict(
            type='YOLACTHead',
            num_classes=80,
            in_channels=256,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                octave_base_scale=3,
                scales_per_octave=1,
                base_sizes=[8, 16, 32, 64, 128],
                ratios=[0.5, 1.0, 2.0],
                strides=[
                    7.971014492753623, 15.714285714285714, 30.555555555555557,
                    61.111111111111114, 110.0
                ],
                centers=[(3.9855072463768115, 3.9855072463768115),
                         (7.857142857142857, 7.857142857142857),
                         (15.277777777777779, 15.277777777777779),
                         (30.555555555555557, 30.555555555555557), (55.0,
                                                                    55.0)]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                reduction='none',
                loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.5),
            num_head_convs=1,
            num_protos=32,
            use_ohem=True),
        mask_head=dict(
            type='YOLACTProtonet',
            in_channels=256,
            num_protos=32,
            num_classes=80,
            max_masks_to_train=100,
            loss_mask_weight=6.125),
        segm_head=dict(
            type='YOLACTSegmHead',
            num_classes=80,
            in_channels=256,
            loss_segm=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
        train_cfg=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0.0,
                ignore_iof_thr=-1,
                gt_max_assign_all=False),
            allowed_border=-1,
            pos_weight=-1,
            neg_pos_ratio=3,
            debug=False),
        test_cfg=dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            iou_thr=0.5,
            top_k=200,
            max_per_img=100)),
    train_cfg=dict(
        use_teacher_proposal=False,
        pseudo_label_initial_score_thr=0.5,
        rpn_pseudo_threshold=0.9,
        cls_pseudo_threshold=0.9,
        reg_pseudo_threshold=0.02,
        jitter_times=10,
        jitter_scale=0.06,
        min_pseduo_box_size=0,
        unsup_weight=4.0),
    test_cfg=dict(inference_on='student'))
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.68, 116.78, 103.94], std=[58.4, 57.12, 57.38], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Sequential',
        transforms=[
            dict(
                type='RandResize',
                img_scale=[(1333, 400), (1333, 1200)],
                multiscale_mode='range',
                keep_ratio=True),
            dict(type='RandFlip', flip_ratio=0.5),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Identity'),
                    dict(type='AutoContrast'),
                    dict(type='RandEqualize'),
                    dict(type='RandSolarize'),
                    dict(type='RandColor'),
                    dict(type='RandContrast'),
                    dict(type='RandBrightness'),
                    dict(type='RandSharpness'),
                    dict(type='RandPosterize')
                ])
        ],
        record=True),
    dict(type='Pad', size_divisor=32),
    dict(
        type='Normalize',
        mean=[123.68, 116.78, 103.94],
        std=[58.4, 57.12, 57.38],
        to_rgb=True),
    dict(type='ExtraAttrs', tag='sup'),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'pad_shape', 'scale_factor', 'tag'))
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.68, 116.78, 103.94],
                std=[58.4, 57.12, 57.38],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=5,
    workers_per_gpu=5,
    train=dict(
        type='SemiDataset',
        sup=dict(
            type='CocoDataset',
            ann_file=
            'data/coco/annotations/semi_supervised/instances_train2017_95.json',
            img_prefix='data/coco/train2017/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
                dict(
                    type='Sequential',
                    transforms=[
                        dict(
                            type='RandResize',
                            img_scale=[(1333, 400), (1333, 1200)],
                            multiscale_mode='range',
                            keep_ratio=True),
                        dict(type='RandFlip', flip_ratio=0.5),
                        dict(
                            type='OneOf',
                            transforms=[
                                dict(type='Identity'),
                                dict(type='AutoContrast'),
                                dict(type='RandEqualize'),
                                dict(type='RandSolarize'),
                                dict(type='RandColor'),
                                dict(type='RandContrast'),
                                dict(type='RandBrightness'),
                                dict(type='RandSharpness'),
                                dict(type='RandPosterize')
                            ])
                    ],
                    record=True),
                dict(type='Pad', size_divisor=32),
                dict(
                    type='Normalize',
                    mean=[123.68, 116.78, 103.94],
                    std=[58.4, 57.12, 57.38],
                    to_rgb=True),
                dict(type='ExtraAttrs', tag='sup'),
                dict(type='DefaultFormatBundle'),
                dict(
                    type='Collect',
                    keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
                    meta_keys=('filename', 'ori_shape', 'img_shape',
                               'img_norm_cfg', 'pad_shape', 'scale_factor',
                               'tag'))
            ]),
        unsup=dict(
            type='CocoDataset',
            ann_file=
            'data/coco/annotations/semi_supervised/instances_train2017_95_unlabeled.json',
            img_prefix='data/coco/train2017/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='PseudoSamples', with_bbox=True, with_mask=True),
                dict(
                    type='MultiBranch',
                    unsup_student=[
                        dict(
                            type='Sequential',
                            transforms=[
                                dict(
                                    type='RandResize',
                                    img_scale=[(1333, 400), (1333, 1200)],
                                    multiscale_mode='range',
                                    keep_ratio=True),
                                dict(type='RandFlip', flip_ratio=0.5),
                                dict(
                                    type='ShuffledSequential',
                                    transforms=[
                                        dict(
                                            type='OneOf',
                                            transforms=[
                                                dict(type='Identity'),
                                                dict(type='AutoContrast'),
                                                dict(type='RandEqualize'),
                                                dict(type='RandSolarize'),
                                                dict(type='RandColor'),
                                                dict(type='RandContrast'),
                                                dict(type='RandBrightness'),
                                                dict(type='RandSharpness'),
                                                dict(type='RandPosterize')
                                            ]),
                                        dict(
                                            type='OneOf',
                                            transforms=[{
                                                'type': 'RandTranslate',
                                                'x': (-0.1, 0.1)
                                            },
                                                        {
                                                            'type':
                                                            'RandTranslate',
                                                            'y': (-0.1, 0.1)
                                                        },
                                                        {
                                                            'type':
                                                            'RandRotate',
                                                            'angle': (-30, 30)
                                                        },
                                                        [{
                                                            'type':
                                                            'RandShear',
                                                            'x': (-30, 30)
                                                        },
                                                         {
                                                             'type':
                                                             'RandShear',
                                                             'y': (-30, 30)
                                                         }]])
                                    ]),
                                dict(
                                    type='RandErase',
                                    n_iterations=(1, 5),
                                    size=[0, 0.2],
                                    squared=True)
                            ],
                            record=True),
                        dict(type='Pad', size_divisor=32),
                        dict(
                            type='Normalize',
                            mean=[123.68, 116.78, 103.94],
                            std=[58.4, 57.12, 57.38],
                            to_rgb=True),
                        dict(type='ExtraAttrs', tag='unsup_student'),
                        dict(type='DefaultFormatBundle'),
                        dict(
                            type='Collect',
                            keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
                            meta_keys=('filename', 'ori_shape', 'img_shape',
                                       'img_norm_cfg', 'pad_shape',
                                       'scale_factor', 'tag',
                                       'transform_matrix'))
                    ],
                    unsup_teacher=[
                        dict(
                            type='Sequential',
                            transforms=[
                                dict(
                                    type='RandResize',
                                    img_scale=[(1333, 400), (1333, 1200)],
                                    multiscale_mode='range',
                                    keep_ratio=True),
                                dict(type='RandFlip', flip_ratio=0.5)
                            ],
                            record=True),
                        dict(type='Pad', size_divisor=32),
                        dict(
                            type='Normalize',
                            mean=[123.68, 116.78, 103.94],
                            std=[58.4, 57.12, 57.38],
                            to_rgb=True),
                        dict(type='ExtraAttrs', tag='unsup_teacher'),
                        dict(type='DefaultFormatBundle'),
                        dict(
                            type='Collect',
                            keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
                            meta_keys=('filename', 'ori_shape', 'img_shape',
                                       'img_norm_cfg', 'pad_shape',
                                       'scale_factor', 'tag',
                                       'transform_matrix'))
                    ])
            ],
            filter_empty_gt=False)),
    val=dict(
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.68, 116.78, 103.94],
                        std=[58.4, 57.12, 57.38],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.68, 116.78, 103.94],
                        std=[58.4, 57.12, 57.38],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    sampler=dict(
        train=dict(
            type='SemiBalanceSampler',
            sample_ratio=[1, 1],
            by_prob=True,
            epoch_length=7330)))
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict()
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    step=[120000, 160000])
runner = dict(type='IterBasedRunner', max_iters=180000)
cudnn_benchmark = True
evaluation = dict(
    metric=['bbox', 'segm'], type='SubModulesDistEvalHook', interval=4000)
mmdet_base = '../../thirdparty/mmdetection/configs'
strong_pipeline = [
    dict(
        type='Sequential',
        transforms=[
            dict(
                type='RandResize',
                img_scale=[(1333, 400), (1333, 1200)],
                multiscale_mode='range',
                keep_ratio=True),
            dict(type='RandFlip', flip_ratio=0.5),
            dict(
                type='ShuffledSequential',
                transforms=[
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(type='Identity'),
                            dict(type='AutoContrast'),
                            dict(type='RandEqualize'),
                            dict(type='RandSolarize'),
                            dict(type='RandColor'),
                            dict(type='RandContrast'),
                            dict(type='RandBrightness'),
                            dict(type='RandSharpness'),
                            dict(type='RandPosterize')
                        ]),
                    dict(
                        type='OneOf',
                        transforms=[{
                            'type': 'RandTranslate',
                            'x': (-0.1, 0.1)
                        }, {
                            'type': 'RandTranslate',
                            'y': (-0.1, 0.1)
                        }, {
                            'type': 'RandRotate',
                            'angle': (-30, 30)
                        },
                                    [{
                                        'type': 'RandShear',
                                        'x': (-30, 30)
                                    }, {
                                        'type': 'RandShear',
                                        'y': (-30, 30)
                                    }]])
                ]),
            dict(
                type='RandErase',
                n_iterations=(1, 5),
                size=[0, 0.2],
                squared=True)
        ],
        record=True),
    dict(type='Pad', size_divisor=32),
    dict(
        type='Normalize',
        mean=[123.68, 116.78, 103.94],
        std=[58.4, 57.12, 57.38],
        to_rgb=True),
    dict(type='ExtraAttrs', tag='unsup_student'),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'pad_shape', 'scale_factor', 'tag', 'transform_matrix'))
]
weak_pipeline = [
    dict(
        type='Sequential',
        transforms=[
            dict(
                type='RandResize',
                img_scale=[(1333, 400), (1333, 1200)],
                multiscale_mode='range',
                keep_ratio=True),
            dict(type='RandFlip', flip_ratio=0.5)
        ],
        record=True),
    dict(type='Pad', size_divisor=32),
    dict(
        type='Normalize',
        mean=[123.68, 116.78, 103.94],
        std=[58.4, 57.12, 57.38],
        to_rgb=True),
    dict(type='ExtraAttrs', tag='unsup_teacher'),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'pad_shape', 'scale_factor', 'tag', 'transform_matrix'))
]
unsup_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='PseudoSamples', with_bbox=True, with_mask=True),
    dict(
        type='MultiBranch',
        unsup_student=[
            dict(
                type='Sequential',
                transforms=[
                    dict(
                        type='RandResize',
                        img_scale=[(1333, 400), (1333, 1200)],
                        multiscale_mode='range',
                        keep_ratio=True),
                    dict(type='RandFlip', flip_ratio=0.5),
                    dict(
                        type='ShuffledSequential',
                        transforms=[
                            dict(
                                type='OneOf',
                                transforms=[
                                    dict(type='Identity'),
                                    dict(type='AutoContrast'),
                                    dict(type='RandEqualize'),
                                    dict(type='RandSolarize'),
                                    dict(type='RandColor'),
                                    dict(type='RandContrast'),
                                    dict(type='RandBrightness'),
                                    dict(type='RandSharpness'),
                                    dict(type='RandPosterize')
                                ]),
                            dict(
                                type='OneOf',
                                transforms=[{
                                    'type': 'RandTranslate',
                                    'x': (-0.1, 0.1)
                                }, {
                                    'type': 'RandTranslate',
                                    'y': (-0.1, 0.1)
                                }, {
                                    'type': 'RandRotate',
                                    'angle': (-30, 30)
                                },
                                            [{
                                                'type': 'RandShear',
                                                'x': (-30, 30)
                                            },
                                             {
                                                 'type': 'RandShear',
                                                 'y': (-30, 30)
                                             }]])
                        ]),
                    dict(
                        type='RandErase',
                        n_iterations=(1, 5),
                        size=[0, 0.2],
                        squared=True)
                ],
                record=True),
            dict(type='Pad', size_divisor=32),
            dict(
                type='Normalize',
                mean=[123.68, 116.78, 103.94],
                std=[58.4, 57.12, 57.38],
                to_rgb=True),
            dict(type='ExtraAttrs', tag='unsup_student'),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
                meta_keys=('filename', 'ori_shape', 'img_shape',
                           'img_norm_cfg', 'pad_shape', 'scale_factor', 'tag',
                           'transform_matrix'))
        ],
        unsup_teacher=[
            dict(
                type='Sequential',
                transforms=[
                    dict(
                        type='RandResize',
                        img_scale=[(1333, 400), (1333, 1200)],
                        multiscale_mode='range',
                        keep_ratio=True),
                    dict(type='RandFlip', flip_ratio=0.5)
                ],
                record=True),
            dict(type='Pad', size_divisor=32),
            dict(
                type='Normalize',
                mean=[123.68, 116.78, 103.94],
                std=[58.4, 57.12, 57.38],
                to_rgb=True),
            dict(type='ExtraAttrs', tag='unsup_teacher'),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
                meta_keys=('filename', 'ori_shape', 'img_shape',
                           'img_norm_cfg', 'pad_shape', 'scale_factor', 'tag',
                           'transform_matrix'))
        ])
]
fp16 = dict(loss_scale='dynamic')
percent = 95
work_dir = 'work_dirs/soft_teacher_yolact_r101_1x8_coco/95'
cfg_name = 'soft_teacher_yolact_r101_1x8_coco'
gpu_ids = range(0, 1)