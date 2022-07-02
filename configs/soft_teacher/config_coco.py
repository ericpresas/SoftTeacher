_base_ = "base.py"

data = dict(
    samples_per_gpu=5,
    workers_per_gpu=0,
    train=dict(
        sup=dict(
            type="CocoDataset",
            ann_file="/mnt/gpid08/users/eric.presas/TFM/datasets/coco_annotations/semi_supervised/instances_train2017.${fold}@${percent}.json",
            img_prefix="/mnt/gpid08/datasets/coco-2017/coco/images/train2017/",
        ),
        unsup=dict(
            type="CocoDataset",
            ann_file="/mnt/gpid08/users/eric.presas/TFM/datasets/coco_annotations/semi_supervised/instances_train2017.${fold}@${percent}-unlabeled.json",
            img_prefix="/mnt/gpid08/datasets/coco-2017/coco/images/train2017/",
        ),
    ),
    val=dict(
        type="CocoDataset",
        ann_file="/mnt/gpid08/users/eric.presas/TFM/datasets/coco_annotations/instances_val2017.json",
        img_prefix="/mnt/gpid08/datasets/coco-2017/coco/images/val2017/",
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 4],
        )
    ),
)

fold = 1
percent = 1

work_dir = "work_dirs/${cfg_name}/${percent}/${fold}"
log_config = dict(
    interval=50,
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
