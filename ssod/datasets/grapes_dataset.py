import mmcv
import numpy as np

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmdet.datasets.coco import CocoDataset


@DATASETS.register_module()
class GrapesDataset(CocoDataset):

    CLASSES = ('grapes')