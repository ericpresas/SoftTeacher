_base_ = "base.py"

data = dict(
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
            sample_ratio=[1, 2],
        )
    ),
)

fold = 1
percent = 1

semi_wrapper = dict(
    type="SoftTeacher",
    model="${model}",
    train_cfg=dict(
        use_teacher_proposal=False,
        pseudo_label_initial_score_thr=0.5,
        rpn_pseudo_threshold=0.75,
        cls_pseudo_threshold=0.75,
        reg_pseudo_threshold=0.02,
        jitter_times=10,
        jitter_scale=0.06,
        min_pseduo_box_size=0,
        unsup_weight=4.0,
    ),
    test_cfg=dict(inference_on="student"),
)

work_dir = "work_dirs/${cfg_name}/${percent}/${fold}"
log_config = dict(
    interval=10,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="pre_release",
                name="${cfg_name}",
                config=dict(
                    fold="${fold}",
                    percent="${percent}",
                    work_dirs="${work_dir}",
                    total_step="${runner.max_iters}",
                ),
            ),
            by_epoch=False,
        ),
    ],
)

optimizer = dict(type="SGD", lr=0.0001, momentum=0.9, weight_decay=0.0001)

semi_wrapper = dict(
    type="SoftTeacher",
    model="${model}",
    train_cfg=dict(
        use_teacher_proposal=True,
        pseudo_label_initial_score_thr=0.5,
        rpn_pseudo_threshold=0.9,
        cls_pseudo_threshold=0.9,
        reg_pseudo_threshold=0.02,
        jitter_times=10,
        jitter_scale=0.06,
        min_pseduo_box_size=0,
        unsup_weight=4.0,
    ),
    test_cfg=dict(inference_on="student"),
)
