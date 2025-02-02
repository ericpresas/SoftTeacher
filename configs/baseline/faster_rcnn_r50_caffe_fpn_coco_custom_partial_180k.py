_base_ = "base.py"
fold = 1
percent = 1
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        classes=("grapes"),
        ann_file="/mnt/gpid08/users/eric.presas/TFM/datasets/grapes_annotations/semi_supervised/instances_train2017.${fold}@${percent}.json",
        img_prefix="/mnt/gpid08/users/eric.presas/TFM/datasets/grapes_dataset",
    ),
)
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
