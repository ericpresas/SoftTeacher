import torch
from torch import nn
from mmcv.runner.fp16_utils import force_fp32
from mmdet.core import bbox2roi, multi_apply
from mmdet.models import DETECTORS, build_detector
device = torch.device('cuda')

from ssod.utils.structure_utils import dict_split, weighted_loss
from ssod.utils import log_image_with_boxes, log_every_n
from ssod.utils.utils import plot_annotations, plot_annotations_xyxy
from ssod.utils.logger import color_transform

from .multi_stream_detector_one_stage import MultiSteamDetector
from .utils import Transform2D, filter_invalid, save_variables_pickle
from mmdet.core import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy

from kornia import augmentation as K
from kornia.augmentation import AugmentationSequential


@DETECTORS.register_module()
class SoftTeacherDETR(MultiSteamDetector):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        super(SoftTeacherDETR, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if train_cfg is not None:
            self.freeze("teacher")
            self.unsup_weight = self.train_cfg.unsup_weight

        self.aug_pipeline = AugmentationSequential(
            K.RandomAffine((-15., 20.), p=1.),
            #data_keys=["input", "bbox_xywh"],
            data_keys=["input", "bbox_xyxy"],
            same_on_batch=False,
            random_apply=1
        )

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
        print(torch.cuda.memory_summary())
        tnames = [meta["filename"] for meta in teacher_data["img_metas"]]
        snames = [meta["filename"] for meta in student_data["img_metas"]]
        tidx = [tnames.index(name) for name in snames]
        with torch.no_grad():
            teacher_info = self.extract_teacher_info(
                teacher_data["img"][
                    torch.Tensor(tidx).to(teacher_data["img"].device).long()
                ],
                [teacher_data["img_metas"][idx] for idx in tidx],
                [teacher_data["proposals"][idx] for idx in tidx]
                if ("proposals" in teacher_data)
                and (teacher_data["proposals"] is not None)
                else None,
            )
        print("Teacher Data ------------------------------------")
        print(teacher_info.keys())
        print("Student Data ------------------------------------")
        print(student_data.keys())
        print("Student gt boxes --------------------------------")
        print(student_data['gt_bboxes']) # Tensors buits

        print(torch.cuda.memory_summary())
        student_info = self.extract_student_info(**student_data)

        return self.compute_pseudo_label_loss(student_info, teacher_info)

    def compute_pseudo_label_loss(self, student_info, teacher_info):
        M = self._get_trans_mat(
            teacher_info["transform_matrix"], student_info["transform_matrix"]
        )

        pseudo_bboxes = self._transform_bbox(
            teacher_info["det_bboxes"],
            M,
            [meta["img_shape"] for meta in student_info["img_metas"]],
        )
        pseudo_labels = teacher_info["det_labels"]
        loss = {}

        loss.update(
            self.unsup_cls_loss(
                pseudo_bboxes,
                pseudo_labels,
                teacher_info=teacher_info,
                student_info=student_info
            )
        )
        """loss.update(
            self.unsup_rcnn_reg_loss(
                student_info["backbone_feature"],
                student_info["img_metas"],
                proposals,
                pseudo_bboxes,
                pseudo_labels,
                student_info=student_info,
            )
        )"""
        return loss

    def unsup_cls_loss(
            self,
            pseudo_bboxes,
            pseudo_labels,
            teacher_info,
            student_info
    ):
        gt_bboxes, gt_labels, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [bbox[:, 4] for bbox in pseudo_bboxes],
            thr=self.train_cfg.cls_pseudo_threshold,
        )

        log_every_n(
            {"cls_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )

        print()

        sampling_results = self.get_sampling_result(
            student_info['img_metas'],
            student_info['det_bboxes'],
            student_info['labels'],
            gt_bboxes,
            gt_labels
        )

        print()

        loss = self.student.bbox_head.loss(
            student_info["cls_scores"],
            student_info["det_bboxes"],
            gt_bboxes,
            gt_labels,
            student_info["img_metas"]
        )

        return loss

    def unsup_rcnn_reg_loss(
        self,
        feat,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        student_info=None,
        **kwargs,
    ):
        gt_bboxes, gt_labels, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [-bbox[:, 5:].mean(dim=-1) for bbox in pseudo_bboxes],
            thr=-self.train_cfg.reg_pseudo_threshold,
        )
        log_every_n(
            {"rcnn_reg_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )
        loss_bbox = self.student.roi_head.forward_train(
            feat, img_metas, proposal_list, gt_bboxes, gt_labels, **kwargs
        )["loss_bbox"]
        if len(gt_bboxes[0]) > 0:
            log_image_with_boxes(
                "rcnn_reg",
                student_info["img"][0],
                gt_bboxes[0],
                bbox_tag="pseudo_label",
                labels=gt_labels[0],
                class_names=self.CLASSES,
                interval=500,
                img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            )
        return {"loss_bbox": loss_bbox}

    def get_sampling_result(
        self,
        img_metas,
        proposal_list,
        proposal_label_list,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        **kwargs,
    ):
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.student.bbox_head.assigner.assign(
                proposal_list[i], proposal_label_list[i], gt_bboxes[i], gt_labels[i], img_metas[i], gt_bboxes_ignore[i]
            )
            sampling_result = self.student.bbox_head.sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
            )
            sampling_results.append(sampling_result)
        return sampling_results

    @force_fp32(apply_to=["bboxes", "trans_mat"])
    def _transform_bbox(self, bboxes, trans_mat, max_shape):
        bboxes = Transform2D.transform_bboxes(bboxes, trans_mat, max_shape)
        return bboxes

    @force_fp32(apply_to=["a", "b"])
    def _get_trans_mat(self, a, b):
        return [bt @ at.inverse() for bt, at in zip(b, a)]

    def extract_student_info(self, img, img_metas, proposals=None, **kwargs):
        student_info = {}
        student_info["img"] = img

        feat = self.student.extract_feat(img)
        student_info["backbone_feature"] = feat

        student_info["img_metas"] = img_metas

        proposal_and_label_list = self.student.bbox_head.simple_test(
            feat, img_metas, rescale=False
        )

        proposal_list = []
        proposal_label_list = []
        for proposal_and_label in proposal_and_label_list:
            proposal_list.append(proposal_and_label[0])
            proposal_label_list.append(proposal_and_label[1])

        proposal_list = [p.to(feat[0].device) for p in proposal_list]
        proposal_list = [
            p if p.shape[0] > 0 else p.new_zeros(0, 5) for p in proposal_list
        ]
        proposal_label_list = [p.to(feat[0].device) for p in proposal_label_list]
        # filter invalid box roughly
        if isinstance(self.train_cfg.pseudo_label_initial_score_thr, float):
            thr = self.train_cfg.pseudo_label_initial_score_thr
        else:
            # TODO: use dynamic threshold
            raise NotImplementedError("Dynamic Threshold is not implemented yet.")

        #________________________
        #cls_scores, proposal_list = self.student.bbox_head(feat, img_metas)
        student_info["det_bboxes"] = proposal_list
        student_info["labels"] = proposal_label_list
        student_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        return student_info

    def extract_teacher_info(self, img, img_metas, proposals=None, **kwargs):
        # For single stage detectors we don't have proposals, so rpn_head output will be ignored.
        # The detection outputs will be computed directly from the image.
        # TODO: Input several auged images and compute deviation.
        print("Extract teacher info -------------------------------")
        teacher_info = {}
        print(img.shape)
        feat = self.teacher.extract_feat(img)
        teacher_info["backbone_feature"] = feat

        proposal_list, proposal_label_list = self.get_bboxes_and_labels(feat, img_metas)

        det_bboxes = proposal_list
        reg_unc = self.compute_uncertanity(img, img_metas, proposal_list, proposal_label_list)

        det_bboxes = [
            torch.cat([bbox, unc], dim=-1) for bbox, unc in zip(det_bboxes, reg_unc)
        ]

        print(proposal_list)
        det_labels = proposal_label_list
        teacher_info["det_bboxes"] = det_bboxes
        teacher_info["det_labels"] = det_labels
        teacher_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        teacher_info["img_metas"] = img_metas
        return teacher_info

    def get_bboxes_and_labels(self, feat, img_metas):
        proposal_and_label_list = self.teacher.bbox_head.simple_test(
            feat, img_metas, rescale=False
        )

        proposal_list = []
        proposal_label_list = []
        for proposal_and_label in proposal_and_label_list:
            proposal_list.append(proposal_and_label[0])
            proposal_label_list.append(proposal_and_label[1])

        proposal_list = [p.to(feat[0].device) for p in proposal_list]
        proposal_list = [
            p if p.shape[0] > 0 else p.new_zeros(0, 5) for p in proposal_list
        ]
        proposal_label_list = [p.to(feat[0].device) for p in proposal_label_list]
        # filter invalid box roughly
        if isinstance(self.train_cfg.pseudo_label_initial_score_thr, float):
            thr = self.train_cfg.pseudo_label_initial_score_thr
        else:
            # TODO: use dynamic threshold
            raise NotImplementedError("Dynamic Threshold is not implemented yet.")

        proposal_list, proposal_label_list, _ = list(
            zip(
                *[
                    filter_invalid(
                        proposal,
                        proposal_label,
                        proposal[:, -1],
                        thr=thr,
                        min_size=self.train_cfg.min_pseduo_box_size,
                    )
                    for proposal, proposal_label in zip(
                        proposal_list, proposal_label_list
                    )
                ]
            )
        )

        return proposal_list, proposal_label_list

    def compute_uncertanity(self, img, img_metas, proposal_list, proposal_label_list, times=10):
        # TODO: No fer per batch, fer imatge a imatge
        batch_size = img.shape[0]
        """plot_annotations_xyxy(color_transform(img[0], **img_metas[0]["img_norm_cfg"]),
                         boxes=[proposal[:4].to('cpu').numpy() for proposal in proposal_list[0]])"""

        box_unc_list = []
        # For each image in batch compute bbox uncertanity for each prediction.
        for i in range(batch_size):
            bboxes_list = [proposal[:4].float() for proposal in proposal_list[i]]

            bboxes = torch.stack(bboxes_list).unsqueeze(dim=0).float()
            img_to_aug = img[i].cpu().float()
            std_boxes_list = []
            shapes_list = []

            # Apply img augmentation and std calculations x times
            for _ in range(times):

                # Call aug pipeline (using kornia)
                aug_res = self.aug_pipeline(img_to_aug / 255., bboxes.cpu())

                """plot_annotations_xyxy(color_transform(aug_res[0][0] * 255., **img_metas[i]["img_norm_cfg"]),
                                      boxes=[proposal[:4].numpy() for proposal in aug_res[1][0]])"""

                # Denormalize image
                aug_img = aug_res[0][0].cuda() * 255.

                # Get predictions for auged image
                feat = self.teacher.extract_feat(aug_img.unsqueeze(dim=0))
                all_cls_scores, all_bbox_preds = self.teacher.bbox_head(feat, [img_metas[i]])

                # Take last decoder layer predictions
                all_cls_scores = all_cls_scores[-1][-1].cuda()
                all_bbox_preds = all_bbox_preds[-1][-1].cuda()

                all_bbox_preds = all_bbox_preds.squeeze(dim=0).to(dtype=torch.float32)

                # Assign predictions to ground truth
                assign_result = self.teacher.bbox_head.assigner.assign(
                    all_bbox_preds,
                    all_cls_scores.squeeze(dim=0).to(dtype=torch.float32),
                    bboxes.squeeze(dim=0), proposal_label_list[i],
                    img_metas[i]
                )

                img_h, img_w, _ = img_metas[i]['img_shape']

                # Factor for normalized bbox
                factor = all_bbox_preds.new_tensor([img_w, img_h, img_w,
                                               img_h]).unsqueeze(0)

                std_boxes = torch.empty(len(bboxes_list), 4).cuda()
                all_preds_boxes = torch.empty(len(bboxes_list), 4).cuda()

                # Calculate std of bbox coordinates respect to the auged gt proposal
                for idx_aug, label in enumerate(assign_result.labels):
                    if label == 0:  # Take only assigned predictions
                        idx_pred = assign_result.gt_inds[idx_aug]
                        pos_gt_bboxes = bbox_xyxy_to_cxcywh(bboxes_list[idx_pred-1])
                        bbox_pred = all_bbox_preds[idx_aug] * factor
                        pred_and_gt_bbox = torch.stack([bbox_pred.squeeze(dim=0), pos_gt_bboxes])
                        all_preds_boxes[idx_pred - 1] = bbox_cxcywh_to_xyxy(bbox_pred.squeeze(dim=0))
                        #std_boxes[idx_pred-1] = torch.std(pred_and_gt_bbox, dim=0)
                        std_boxes[idx_pred - 1] = pred_and_gt_bbox.std(dim=0)

                # Compute shapes for predictions (later used to compute relative uncertanity)
                shapes_list.append((all_preds_boxes[:, 2:4] - all_preds_boxes[:, :2]).clamp(min=1.0))
                std_boxes_list.append(std_boxes)

            # TODO: Revisar shapes
            # Mean calc of std across all the same assigned prediction over auged images
            # Calc mean across shapes of assigned predictions in auged images
            std_boxes_stack = torch.stack(std_boxes_list)
            std_boxes_stack = torch.mean(std_boxes_stack, dim=0)
            shapes_stack = torch.stack(shapes_list)
            shapes_stack = torch.mean(shapes_stack, dim=0)

            # Relative unc
            if shapes_stack.numel() > 0:
                box_unc = std_boxes_stack / shapes_stack[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
            else:
                box_unc = std_boxes_stack

            box_unc_list.append(box_unc)

        return box_unc_list



    def compute_uncertainty_with_aug(
        self, feat, img_metas, proposal_list, proposal_label_list
    ):
        auged_proposal_list = self.aug_box(
            proposal_list, self.train_cfg.jitter_times, self.train_cfg.jitter_scale
        )
        # flatten
        auged_proposal_list = [
            auged.reshape(-1, auged.shape[-1])[:,:4] for auged in auged_proposal_list
        ]

        """bboxes, _ = self.teacher.roi_head.simple_test_bboxes(
            feat,
            img_metas,
            auged_proposal_list,
            None,
            rescale=False,
        )"""

        bboxes = auged_proposal_list

        reg_channel = max([bbox.shape[-1] for bbox in bboxes]) // 4
        bboxes = [
            bbox.reshape(self.train_cfg.jitter_times, -1, bbox.shape[-1])
            if bbox.numel() > 0
            else bbox.new_zeros(self.train_cfg.jitter_times, 0, 4 * reg_channel).float()
            for bbox in bboxes
        ]

        box_unc = [bbox.std(dim=0) for bbox in bboxes]
        bboxes = [bbox.mean(dim=0) for bbox in bboxes]
        # scores = [score.mean(dim=0) for score in scores]
        if reg_channel != 1:
            bboxes = [
                bbox.reshape(bbox.shape[0], reg_channel, 4)[
                    torch.arange(bbox.shape[0]), label
                ]
                for bbox, label in zip(bboxes, proposal_label_list)
            ]
            box_unc = [
                unc.reshape(unc.shape[0], reg_channel, 4)[
                    torch.arange(unc.shape[0]), label
                ]
                for unc, label in zip(box_unc, proposal_label_list)
            ]

        box_shape = [(bbox[:, 2:4] - bbox[:, :2]).clamp(min=1.0) for bbox in bboxes]
        # relative unc
        box_unc = [
            unc / wh[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
            if wh.numel() > 0
            else unc
            for unc, wh in zip(box_unc, box_shape)
        ]
        return box_unc

    @staticmethod
    def aug_box(boxes, times=1, frac=0.06):
        def _aug_single(box):
            # random translate
            # TODO: random flip or something
            box_scale = box[:, 2:4] - box[:, :2]
            box_scale = (
                box_scale.clamp(min=1)[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
            )
            aug_scale = box_scale * frac  # [n,4]

            offset = (
                torch.randn(times, box.shape[0], 4, device=box.device)
                * aug_scale[None, ...]
            )
            new_box = box.clone()[None, ...].expand(times, box.shape[0], -1)
            return torch.cat(
                [new_box[:, :, :4].clone() + offset, new_box[:, :, 4:]], dim=-1
            )

        return [_aug_single(box) for box in boxes]

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
