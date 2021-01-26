import numpy as np
import torch
import torch.nn.functional as F

from ..utils import multi_apply
from .transforms import bbox2delta, delta2bbox


def bbox_target(
    pos_bboxes_list,
    neg_bboxes_list,
    pos_gt_bboxes_list,
    pos_gt_labels_list,
    cfg,
    reg_classes=1,
    target_means=[0.0, 0.0, 0.0, 0.0],
    target_stds=[1.0, 1.0, 1.0, 1.0],
    concat=True,
):
    labels, label_weights, bbox_targets, bbox_weights = multi_apply(
        bbox_target_single,
        pos_bboxes_list,
        neg_bboxes_list,
        pos_gt_bboxes_list,
        pos_gt_labels_list,
        cfg=cfg,
        reg_classes=reg_classes,
        target_means=target_means,
        target_stds=target_stds,
    )

    if concat:
        labels = torch.cat(labels, 0)
        label_weights = torch.cat(label_weights, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        bbox_weights = torch.cat(bbox_weights, 0)
    return labels, label_weights, bbox_targets, bbox_weights


def bbox_target_single(
    pos_bboxes,
    neg_bboxes,
    pos_gt_bboxes,
    pos_gt_labels,
    cfg,
    reg_classes=1,
    target_means=[0.0, 0.0, 0.0, 0.0],
    target_stds=[1.0, 1.0, 1.0, 1.0],
):
    num_pos = pos_bboxes.size(0)
    num_neg = neg_bboxes.size(0)
    num_samples = num_pos + num_neg
    labels = pos_bboxes.new_zeros(num_samples, dtype=torch.long)
    label_weights = pos_bboxes.new_zeros(num_samples)
    bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
    bbox_weights = pos_bboxes.new_zeros(num_samples, 4)

    if num_pos > 0:
        labels[:num_pos] = pos_gt_labels
        pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
        label_weights[:num_pos] = pos_weight
        pos_bbox_targets = bbox2delta(
            pos_bboxes, pos_gt_bboxes, target_means, target_stds
        )
        bbox_targets[:num_pos, :] = pos_bbox_targets
        bbox_weights[:num_pos, :] = 1
    if num_neg > 0:
        label_weights[-num_neg:] = 1.0

    return labels, label_weights, bbox_targets, bbox_weights


def bbox_target_tsd(
    pos_bboxes_list,
    neg_bboxes_list,
    pos_gt_bboxes_list,
    pos_gt_labels_list,
    rois,
    delta_c,
    delta_r,
    cls_score_,
    bbox_pred_,
    TSD_cls_score_,
    TSD_bbox_pred_,
    cfg,
    reg_classes=1,
    cls_pc_margin=0.2,
    loc_pc_margin=0.2,
    target_means=[0.0, 0.0, 0.0, 0.0],
    target_stds=[1.0, 1.0, 1.0, 1.0],
    concat=True,
):
    labels, label_weights, bbox_targets, bbox_weights, TSD_labels, TSD_label_weights, TSD_bbox_targets, TSD_bbox_weights, pc_cls_loss, pc_loc_loss = multi_apply(
        bbox_target_single_tsd,
        pos_bboxes_list,
        neg_bboxes_list,
        pos_gt_bboxes_list,
        pos_gt_labels_list,
        rois,
        delta_c,
        delta_r,
        cls_score_,
        bbox_pred_,
        TSD_cls_score_,
        TSD_bbox_pred_,
        cfg=cfg,
        reg_classes=reg_classes,
        cls_pc_margin=cls_pc_margin,
        loc_pc_margin=loc_pc_margin,
        target_means=target_means,
        target_stds=target_stds,
    )

    if concat:
        labels = torch.cat(labels, 0)
        label_weights = torch.cat(label_weights, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        bbox_weights = torch.cat(bbox_weights, 0)

        TSD_labels = torch.cat(TSD_labels, 0)
        TSD_label_weights = torch.cat(TSD_label_weights, 0)
        TSD_bbox_targets = torch.cat(TSD_bbox_targets, 0)
        TSD_bbox_weights = torch.cat(TSD_bbox_weights, 0)

        pc_cls_loss = torch.cat(pc_cls_loss, 0)
        pc_loc_loss = torch.cat(pc_loc_loss, 0)

    return (
        labels,
        label_weights,
        bbox_targets,
        bbox_weights,
        TSD_labels,
        TSD_label_weights,
        TSD_bbox_targets,
        TSD_bbox_weights,
        pc_cls_loss,
        pc_loc_loss,
    )


def iou_overlaps(b1, b2):
    """
        Arguments:
            b1: dts, [n, >=4] (x1, y1, x2, y2, ...)
            b1: gts, [n, >=4] (x1, y1, x2, y2, ...)

        Returns:
            intersection-over-union pair-wise, generalized iou.
        """
    area1 = (b1[:, 2] - b1[:, 0] + 1) * (b1[:, 3] - b1[:, 1] + 1)
    area2 = (b2[:, 2] - b2[:, 0] + 1) * (b2[:, 3] - b2[:, 1] + 1)
    # only for giou loss
    lt1 = torch.max(b1[:, :2], b2[:, :2])
    rb1 = torch.max(b1[:, 2:4], b2[:, 2:4])
    lt2 = torch.min(b1[:, :2], b2[:, :2])
    rb2 = torch.min(b1[:, 2:4], b2[:, 2:4])
    wh1 = (rb2 - lt1 + 1).clamp(min=0)
    wh2 = (rb1 - lt2 + 1).clamp(min=0)
    inter_area = wh1[:, 0] * wh1[:, 1]
    union_area = area1 + area2 - inter_area
    iou = inter_area / torch.clamp(union_area, min=1)
    ac_union = wh2[:, 0] * wh2[:, 1] + 1e-7
    giou = iou - (ac_union - union_area) / ac_union
    return iou, giou


def bbox_target_single_tsd(
    pos_bboxes,
    neg_bboxes,
    pos_gt_bboxes,
    pos_gt_labels,
    rois,
    delta_c,
    delta_r,
    cls_score_,
    bbox_pred_,
    TSD_cls_score_,
    TSD_bbox_pred_,
    cfg,
    reg_classes=1,
    cls_pc_margin=0.2,
    loc_pc_margin=0.2,
    target_means=[0.0, 0.0, 0.0, 0.0],
    target_stds=[1.0, 1.0, 1.0, 1.0],
):
    num_pos = pos_bboxes.size(0)
    num_neg = neg_bboxes.size(0)
    num_samples = num_pos + num_neg
    labels = pos_bboxes.new_zeros(num_samples, dtype=torch.long)
    label_weights = pos_bboxes.new_zeros(num_samples)
    bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
    bbox_weights = pos_bboxes.new_zeros(num_samples, 4)

    TSD_labels = pos_bboxes.new_zeros(num_samples, dtype=torch.long)
    TSD_label_weights = pos_bboxes.new_zeros(num_samples)
    TSD_bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
    TSD_bbox_weights = pos_bboxes.new_zeros(num_samples, 4)

    # generte P_r according to delta_r and rois
    w = rois[:, 3] - rois[:, 1] + 1
    h = rois[:, 4] - rois[:, 2] + 1
    scale = 0.1
    rois_r = rois.new_zeros(rois.shape[0], rois.shape[1])
    rois_r[:, 0] = rois[:, 0]
    rois_r[:, 1] = rois[:, 1] + delta_r[:, 0] * scale * w
    rois_r[:, 2] = rois[:, 2] + delta_r[:, 1] * scale * h
    rois_r[:, 3] = rois[:, 3] + delta_r[:, 0] * scale * w
    rois_r[:, 4] = rois[:, 4] + delta_r[:, 1] * scale * h
    TSD_pos_rois = rois_r[:num_pos]
    pos_rois = rois[:num_pos]
    pc_cls_loss = rois.new_zeros(1)
    pc_loc_loss = rois.new_zeros(1)
    if num_pos > 0:
        labels[:num_pos] = pos_gt_labels
        TSD_labels[:num_pos] = pos_gt_labels
        pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
        label_weights[:num_pos] = pos_weight
        TSD_label_weights[:num_pos] = pos_weight
        pos_bbox_targets = bbox2delta(
            pos_bboxes, pos_gt_bboxes, target_means, target_stds
        )
        TSD_pos_bbox_targets = bbox2delta(
            TSD_pos_rois[:, 1:], pos_gt_bboxes, target_means, target_stds
        )
        bbox_targets[:num_pos, :] = pos_bbox_targets
        bbox_weights[:num_pos, :] = 1
        TSD_bbox_targets[:num_pos, :] = TSD_pos_bbox_targets
        TSD_bbox_weights[:num_pos, :] = 1

        # compute PC for TSD
        # 1. compute the PC for classification
        cls_score_soft = F.softmax(cls_score_, dim=1)
        TSD_cls_score_soft = F.softmax(TSD_cls_score_, dim=1)
        cls_pc_margin = (
            torch.tensor(cls_pc_margin).to(labels.device).to(dtype=cls_score_soft.dtype)
        )
        cls_pc_margin = torch.min(
            1 - cls_score_soft[np.arange(len(TSD_labels)), labels], cls_pc_margin
        ).detach()
        pc_cls_loss = F.relu(
            -(
                TSD_cls_score_soft[np.arange(len(TSD_labels)), TSD_labels]
                - cls_score_soft[np.arange(len(TSD_labels)), labels].detach()
                - cls_pc_margin
            )
        )

        # 2. compute the PC for localization
        N = bbox_pred_.shape[0]
        bbox_pred_ = bbox_pred_.view(N, -1, 4)
        TSD_bbox_pred_ = TSD_bbox_pred_.view(N, -1, 4)

        sibling_head_bboxes = delta2bbox(
            pos_bboxes,
            bbox_pred_[np.arange(num_pos), labels[:num_pos]],
            means=target_means,
            stds=target_stds,
        )
        TSD_head_bboxes = delta2bbox(
            TSD_pos_rois[:, 1:],
            TSD_bbox_pred_[np.arange(num_pos), TSD_labels[:num_pos]],
            means=target_means,
            stds=target_stds,
        )

        ious, gious = iou_overlaps(sibling_head_bboxes, pos_gt_bboxes)
        TSD_ious, TSD_gious = iou_overlaps(TSD_head_bboxes, pos_gt_bboxes)
        loc_pc_margin = torch.tensor(loc_pc_margin).to(ious.device).to(dtype=ious.dtype)
        loc_pc_margin = torch.min(1 - ious.detach(), loc_pc_margin).detach()
        pc_loc_loss = F.relu(-(TSD_ious - ious.detach() - loc_pc_margin))

    if num_neg > 0:
        label_weights[-num_neg:] = 1.0
        TSD_label_weights[-num_neg:] = 1.0

    return (
        labels,
        label_weights,
        bbox_targets,
        bbox_weights,
        TSD_labels,
        TSD_label_weights,
        TSD_bbox_targets,
        TSD_bbox_weights,
        pc_cls_loss,
        pc_loc_loss,
    )


def expand_target(bbox_targets, bbox_weights, labels, num_classes):
    bbox_targets_expand = bbox_targets.new_zeros(
        (bbox_targets.size(0), 4 * num_classes)
    )
    bbox_weights_expand = bbox_weights.new_zeros(
        (bbox_weights.size(0), 4 * num_classes)
    )
    for i in torch.nonzero(labels > 0).squeeze(-1):
        start, end = labels[i] * 4, (labels[i] + 1) * 4
        bbox_targets_expand[i, start:end] = bbox_targets[i, :]
        bbox_weights_expand[i, start:end] = bbox_weights[i, :]
    return bbox_targets_expand, bbox_weights_expand
