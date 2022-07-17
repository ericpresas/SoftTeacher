_base_="base_detr.py"

"""data = dict(
    samples_per_gpu=5,
    workers_per_gpu=0,
    train=dict(
        sup=dict(
            type="GrapesDataset",
            classes=["grapes"],
            ann_file="/mnt/gpid08/users/eric.presas/TFM/datasets/grapes_annotations/grapes_train_anns.json",
            img_prefix="/mnt/gpid08/users/eric.presas/TFM/datasets/grapes_dataset",
        ),
        unsup=dict(
            type="GrapesDataset",
            classes=["grapes"],
            ann_file="/mnt/gpid08/users/eric.presas/TFM/datasets/grapes_annotations/grapes_no_anns.json",
            img_prefix="/mnt/gpid08/users/eric.presas/TFM/datasets/grapes_dataset",
        )
    ),
    val=dict(
        type="GrapesDataset",
        classes=["grapes"],
        ann_file="/mnt/gpid08/users/eric.presas/TFM/datasets/grapes_annotations/grapes_val_anns.json",
        img_prefix="/mnt/gpid08/users/eric.presas/TFM/datasets/grapes_dataset"
    ),
    test=dict(
        type="GrapesDataset",
        classes=["grapes"],
        ann_file="/mnt/gpid08/users/eric.presas/TFM/datasets/grapes_annotations/grapes_test_anns.json",
        img_prefix="/mnt/gpid08/users/eric.presas/TFM/datasets/grapes_dataset"
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 1],
        )
    ),
)"""

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        sup=dict(
            type="GrapesDataset",
            classes=["grapes"],
            ann_file="/home/eric/Documents/Datasets/grapes_annotations/grapes_train_anns.json",
            img_prefix="/home/eric/Documents/Datasets/grapes_dataset",
        ),
        unsup=dict(
            type="GrapesDataset",
            classes=["grapes"],
            ann_file="/home/eric/Documents/Datasets/grapes_annotations/grapes_no_anns.json",
            img_prefix="/home/eric/Documents/Datasets/grapes_dataset",
        )
    ),
    val=dict(
        type="GrapesDataset",
        classes=["grapes"],
        ann_file="/home/eric/Documents/Datasets/grapes_annotations/grapes_val_anns.json",
        img_prefix="/home/eric/Documents/Datasets/grapes_dataset"
    ),
    test=dict(
        type="GrapesDataset",
        classes=["grapes"],
        ann_file="/home/eric/Documents/Datasets/grapes_annotations/grapes_test_anns.json",
        img_prefix="/home/eric/Documents/Datasets/grapes_dataset"
    ),
    sampler=dict(
        train=dict(
            _delete_=True,
            type="GroupSampler"
        )
    ),
)

lr_config = dict(step=[120000 * 4, 160000 * 4])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=180000 * 4)

semi_wrapper = dict(
    type='SoftTeacherDETRSimple',
    model="${model}",
    train_cfg=dict(
        use_teacher_proposal=False,
        pseudo_label_initial_score_thr=0.5,
        rpn_pseudo_threshold=0.9,
        cls_pseudo_threshold=0.7,
        reg_pseudo_threshold=0.02,
        jitter_times=10,
        jitter_scale=0.06,
        min_pseduo_box_size=0,
        unsup_weight=2.0,
        load_checkpoint='/home/eric/Documents/Projects/SoftTeacher/work_dirs/detr_check/epoch_409.pth'
    ),
    test_cfg=dict(inference_on='student')
)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=0.1, norm_type=2))
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=180000)
lr_config = dict(step=[120000, 160000])

#load_from = '/home/eric/Documents/Projects/SoftTeacher/work_dirs/detr_check/epoch_409.pth'