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
    samples_per_gpu=2,
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
            sample_ratio=[1, 1],
        )
    ),
)

lr_config = dict(step=[120000 * 4, 160000 * 4])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=180000 * 4)

semi_wrapper = dict(
    type='SoftTeacherDETR',
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
        unsup_weight=2.0),
    test_cfg=dict(inference_on='student')
)

optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=180000)
lr_config = dict(step=[120000, 160000])