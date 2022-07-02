#!/usr/bin/env bash
set -x

TYPE=baseline
FOLD=1
PERCENT=5
GPUS=1
PORT=${PORT:-29500}

export PYTHONPATH=/mnt/gpid08/users/eric.presas/TFM/SoftTeacher/:$PYTHONPATH
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        tools/train.py configs/soft_teacher/config_grapes.py --launcher pytorch \
        --cfg-options fold=${FOLD} percent=${PERCENT}