import torch
from torch import nn
from mmcv.runner.fp16_utils import force_fp32
from mmdet.core import multi_apply
from mmdet.models import DETECTORS, build_detector
device = torch.device('cuda')

from ssod.utils.structure_utils import dict_split, weighted_loss
from ssod.utils import log_image_with_boxes, log_every_n

from .multi_stream_detector_one_stage import MultiSteamDetector
from .utils import Transform2D, filter_invalid


@DETECTORS.register_module()
class SoftTeacherDETRSimple(MultiSteamDetector):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        super(SoftTeacherDETRSimple, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if train_cfg is not None:
            self.freeze("teacher")
            self.unsup_weight = self.train_cfg.unsup_weight

    def forward_train(self, img, img_metas, **kwargs):
        super().forward_train(img, img_metas, **kwargs)
        kwargs.update({"img": img})
        kwargs.update({"img_metas": img_metas})
        kwargs.update({"tag": [meta["tag"] for meta in img_metas]})
        data_groups = dict_split(kwargs, "tag")
        for _, v in data_groups.items():
            v.pop("tag")

        loss = {}
        #! Warnings: By splitting losses for supervised data and unsupervised data with different names,
        #! it means that at least one sample for each group should be provided on each gpu.
        #! In some situation, we can only put one image per gpu, we have to return the sum of loss
        #! and log the loss with logger instead. Or it will try to sync tensors don't exist.
        if "sup" in data_groups:
            gt_bboxes = data_groups["sup"]["gt_bboxes"]
            log_every_n(
                {"sup_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
            )
            sup_loss = self.student.forward_train(**data_groups["sup"])
            sup_loss = {"sup_" + k: v for k, v in sup_loss.items()}
            loss.update(**sup_loss)
        if "unsup_student" in data_groups:
            unsup_loss = weighted_loss(
                self.foward_unsup_train(
                    data_groups["unsup_teacher"], data_groups["unsup_student"]
                ),
                weight=self.unsup_weight,
            )
            unsup_loss = {"unsup_" + k: v for k, v in unsup_loss.items()}
            loss.update(**unsup_loss)

        return loss

    def foward_unsup_train(self, teacher_data, student_data):
        # sort the teacher and student input to avoid some bugs
        tnames = [meta["filename"] for meta in teacher_data["img_metas"]]
        snames = [meta["filename"] for meta in student_data["img_metas"]]
        tidx = [tnames.index(name) for name in snames]
        with torch.no_grad():
            teacher_info = self.extract_teacher_info(
                teacher_data["img"][
                    torch.Tensor(tidx).to(teacher_data["img"].device).long()
                ],
                [teacher_data["img_metas"][idx] for idx in tidx]
            )

        student_info = self.extract_student_info(**student_data)

        return self.compute_pseudo_label_loss(student_info, teacher_info)

    def compute_pseudo_label_loss(self, student_info, teacher_info):  # compute_pseudo_label_loss
        M = self._get_trans_mat(
            teacher_info["transform_matrix"], student_info["transform_matrix"]
        )

        pseudo_bboxes = self._transform_bbox(
            teacher_info["det_bboxes"],
            M,
            [meta["img_shape"] for meta in student_info["img_metas"]],
        )

        pseudo_labels = teacher_info["det_labels"]

        pseudo_bboxes = torch.unsqueeze(pseudo_bboxes, dim=0)

        gt_bboxes, gt_labels, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [bbox[:, 4] for bbox in pseudo_bboxes],
            thr=self.train_cfg.cls_pseudo_threshold,
        )

        loss = self.student.bbox_head.forward_train(
            list(student_info["backbone_feature"]),
            img_metas=student_info["img_metas"],
            gt_bboxes=torch.stack(gt_bboxes),
            gt_labels=torch.stack(gt_labels)
        )
        return loss

    @force_fp32(apply_to=["bboxes", "trans_mat"])
    def _transform_bbox(self, bboxes, trans_mat, max_shape):
        bboxes = Transform2D.transform_bboxes(torch.squeeze(bboxes), trans_mat[0], max_shape[0])
        return bboxes

    @force_fp32(apply_to=["a", "b"])
    def _get_trans_mat(self, a, b):
        return [bt @ at.inverse() for bt, at in zip(b, a)]

    def extract_student_info(self, img, img_metas, **kwargs):
        student_info = {}
        student_info["img"] = img
        feat = self.student.extract_feat(img)
        student_info["backbone_feature"] = feat
        student_info["img_metas"] = img_metas

        student_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        return student_info

    def extract_teacher_info(self, img, img_metas, **kwargs):
        teacher_info = {}
        feat = self.teacher.extract_feat(img)
        teacher_info["backbone_feature"] = feat

        outs = self.teacher.bbox_head(feat, img_metas)
        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:].to(device)
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape

        """if len(outs) == 2:
            # add dummy score_factor
            outs = (*outs, None)"""
        det_bboxes, det_labels = self.teacher.bbox_head.onnx_export(
            *outs, img_metas)

        teacher_info["det_bboxes"] = det_bboxes
        teacher_info["det_labels"] = det_labels
        teacher_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        teacher_info["img_metas"] = img_metas
        return teacher_info

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if not any(["student" in key or "teacher" in key for key in state_dict.keys()]):
            keys = list(state_dict.keys())
            state_dict.update({"teacher." + k: state_dict[k] for k in keys})
            state_dict.update({"student." + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )