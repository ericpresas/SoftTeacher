_base_="base.py"

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

semi_wrapper = dict(
    train_cfg=dict(
        unsup_weight=2.0,
    )
)

lr_config = dict(step=[120000 * 4, 160000 * 4])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=180000 * 4)

