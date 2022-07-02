_base_ = "base.py"

"""data = dict(
    samples_per_gpu=7,
    workers_per_gpu=0,
    train=dict(
        sup=dict(
            type="GrapesDataset",
            ann_file="/mnt/gpid08/users/eric.presas/TFM/datasets/grapes_annotations/semi_supervised/instances_train2017.1@10.json",
            img_prefix="/mnt/gpid08/users/eric.presas/TFM/datasets/grapes_dataset",
        ),
        unsup=dict(
            type="GrapesDataset",
            ann_file="/mnt/gpid08/users/eric.presas/TFM/datasets/grapes_annotations/semi_supervised/instances_train2017.1@10-unlabeled.json",
            img_prefix="/mnt/gpid08/users/eric.presas/TFM/datasets/grapes_dataset",
        )
    ),
    val=dict(
        type="GrapesDataset",
        ann_file="/mnt/gpid08/users/eric.presas/TFM/datasets/grapes_annotations/grapes_val_anns.json",
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

fold = 1
percent = 1

work_dir = "work_dirs/${cfg_name}/${percent}/${fold}"
"""log_config = dict(
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
)"""

log_config = dict(
    interval=10,
    hooks=[
        dict(type="TextLoggerHook")
    ],
)
