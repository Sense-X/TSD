import torch
import torch.nn as nn
import numpy as np

from mmdet.ops import ConvModule
from ..registry import HEADS
from .bbox_head import BBoxHead
from mmdet.ops import DeltaRPooling, DeltaCPooling
from mmdet.core import (auto_fp16, bbox_target_tsd, delta2bbox, force_fp32,
                        multiclass_nms)
from ..losses import accuracy

@HEADS.register_module
class TSDConvFCBBoxHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

                                /-> cls convs -> cls fcs -> cls
    shared convs -> shared fcs
                                \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 cls_pc_margin=0.2,
                 loc_pc_margin=0.2,
                 featmap_strides = None,
                 *args,
                 **kwargs):
        super(TSDConvFCBBoxHead, self).__init__(*args, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.cls_pc_margin = cls_pc_margin
        self.loc_pc_margin = loc_pc_margin
        # add shared fc and specific fcs to generate delta_c and delta_r for disentangling input proposals
        self.shared_fc = nn.Sequential(
            nn.Linear(self.roi_feat_area * self.in_channels, 256),
            nn.ReLU(inplace=True))
        self.delta_c = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.roi_feat_area * 2))
        self.delta_r = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2))

        # add AplignPool for Pc and Pr
        self.pool_size = int(np.sqrt(self.roi_feat_area))
        self.align_pooling_pc = nn.ModuleList([DeltaCPooling(spatial_scale=1.0 / x,
            out_size=self.pool_size,
            out_channels=self.in_channels,
            no_trans=False,
            group_size=1,
            trans_std=0.1) for x in featmap_strides])
        self.align_pooling_pr = nn.ModuleList([DeltaRPooling(spatial_scale=1.0 / x,
            out_size=self.pool_size,
            out_channels=self.in_channels,
            no_trans=False,
            group_size=1,
            trans_std=0.1) for x in featmap_strides])

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add TSD convs and fcs
        self.TSD_pc_convs, self.TSD_pc_fcs, TSD_last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels, True)
        self.TSD_pr_convs, self.TSD_pr_fcs, TSD_last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels, True)
        self.TSD_out_channels = TSD_last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add TSD cls specific branch
        self.TSD_cls_convs, self.TSD_cls_fcs, self.TSD_cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.TSD_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        # add TSD reg specific branch
        self.TSD_reg_convs, self.TSD_reg_fcs, self.TSD_reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.TSD_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
                self.TSD_cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area
                self.TSD_reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes)
            self.TSD_fc_cls = nn.Linear(self.TSD_cls_last_dim, self.num_classes)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)
            self.TSD_fc_reg = nn.Linear(self.TSD_reg_last_dim, out_dim_reg)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super(TSDConvFCBBoxHead, self).init_weights()
        # conv layers are already initialized by ConvModule
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs, self.TSD_pc_fcs, self.TSD_pr_fcs, self.TSD_cls_fcs, self.TSD_reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    # nn.init.xavier_uniform_(m.weight)
                    nn.init.kaiming_normal_(m.weight.data, a=1)
                    nn.init.constant_(m.bias, 0)

        for module_list in [self.shared_fc, self.delta_c, self.delta_r]:
            for m in module_list.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight.data, a=1)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        finest_scale=56
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1] + 1) * (rois[:, 4] - rois[:, 2] + 1))
        target_lvls = torch.floor(torch.log2(scale / finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    @force_fp32(apply_to=('feats'))
    def forward(self, x, feats, rois):
        # generate TSD pc pr and corresponding features
        c = x.numel() // x.shape[0]
        x1 = x.view(-1, c)
        x2= self.shared_fc(x1)
        delta_c = self.delta_c(x2)
        delta_r = self.delta_r(x2)
        num_levels = len(feats)
        target_lvls = self.map_roi_levels(rois, num_levels)
        TSD_cls_feats = x.new_zeros(
            rois.size(0), self.in_channels, self.pool_size, self.pool_size)
        TSD_loc_feats = x.new_zeros(
            rois.size(0), self.in_channels, self.pool_size, self.pool_size)
        for i in range(num_levels):
            inds = target_lvls == i
            if inds.any():
                delta_c_ = delta_c[inds, :]
                delta_r_ = delta_r[inds, :]
                rois_ = rois[inds, :]
                tsd_feats_cls = self.align_pooling_pc[i](feats[i], rois_, delta_c_.to(dtype=rois_.dtype))
                tsd_feats_loc = self.align_pooling_pr[i](feats[i], rois_, delta_r_.to(dtype=rois_.dtype))
                TSD_cls_feats[inds] = tsd_feats_cls.to(dtype=x.dtype)
                TSD_loc_feats[inds] = tsd_feats_loc.to(dtype=x.dtype)

        # shared part for TSD
        if self.num_shared_convs > 0:
            for conv in self.TSD_pc_convs:
                TSD_cls_feats = conv(TSD_cls_feats)
            for conv in self.TSD_pr_convs:
                TSD_loc_feats = conv(TSD_loc_feats)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                TSD_cls_feats = self.avg_pool(TSD_cls_feats)
                TSD_loc_feats = self.avg_pool(TSD_loc_feats)

            TSD_cls_feats = TSD_cls_feats.flatten(1)
            TSD_loc_feats = TSD_loc_feats.flatten(1)

            for fc in self.TSD_pc_fcs:
                TSD_cls_feats = self.relu(fc(TSD_cls_feats))
            for fc in self.TSD_pr_fcs:
                TSD_loc_feats = self.relu(fc(TSD_loc_feats))
            # separate branches
        TSD_x_cls = TSD_cls_feats
        TSD_x_reg = TSD_loc_feats
        for conv in self.TSD_cls_convs:
            TSD_x_cls = conv(TSD_x_cls)
        if TSD_x_cls.dim() > 2:
            if self.with_avg_pool:
                TSD_x_cls = self.avg_pool(TSD_x_cls)
            TSD_x_cls = TSD_x_cls.flatten(1)
        for fc in self.TSD_cls_fcs:
            TSD_x_cls = self.relu(fc(TSD_x_cls))

        for conv in self.TSD_reg_convs:
            TSD_x_reg = conv(TSD_x_reg)
        if TSD_x_reg.dim() > 2:
            if self.with_avg_pool:
                TSD_x_reg = self.avg_pool(TSD_x_reg)
                TSD_x_reg = TSD_x_reg.flatten(1)
        for fc in self.TSD_reg_fcs:
            TSD_x_reg = self.relu(fc(TSD_x_reg))

        TSD_cls_score = self.TSD_fc_cls(TSD_x_cls) if self.with_cls else None
        TSD_bbox_pred = self.TSD_fc_reg(TSD_x_reg) if self.with_reg else None

        # shared part for sibling head, only used in training phase.
        if self.training:
            if self.num_shared_convs > 0:
                for conv in self.shared_convs:
                    x = conv(x)

            if self.num_shared_fcs > 0:
                if self.with_avg_pool:
                    x = self.avg_pool(x)

                x = x.flatten(1)

                for fc in self.shared_fcs:
                    x = self.relu(fc(x))
            # separate branches
            x_cls = x
            x_reg = x

            for conv in self.cls_convs:
                x_cls = conv(x_cls)
            if x_cls.dim() > 2:
                if self.with_avg_pool:
                    x_cls = self.avg_pool(x_cls)
                x_cls = x_cls.flatten(1)
            for fc in self.cls_fcs:
                x_cls = self.relu(fc(x_cls))

            for conv in self.reg_convs:
                x_reg = conv(x_reg)
            if x_reg.dim() > 2:
                if self.with_avg_pool:
                    x_reg = self.avg_pool(x_reg)
                x_reg = x_reg.flatten(1)
            for fc in self.reg_fcs:
                x_reg = self.relu(fc(x_reg))

            cls_score = self.fc_cls(x_cls) if self.with_cls else None
            bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
            return cls_score, bbox_pred, TSD_cls_score, TSD_bbox_pred, delta_c, delta_r
        else:
            return None, None, TSD_cls_score, TSD_bbox_pred, delta_c, delta_r

    @force_fp32(apply_to=('delta_c', 'delta_r', 'TSD_cls_score', 'TSD_bbox_pred', 'cls_score', 'bbox_pred'))
    def get_target(self, rois, sampling_results, gt_bboxes, gt_labels, delta_c, delta_r, cls_score, bbox_pred, TSD_cls_score, TSD_bbox_pred,
                   rcnn_train_cfg, img_metas):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        
        rois_ = [rois[(rois[:,0]==i).type(torch.bool)] for i in range(len(sampling_results))]
        delta_c_ = [delta_c[(rois[:,0]==i).type(torch.bool)] for i in range(len(sampling_results))]
        delta_r_ = [delta_r[(rois[:,0]==i).type(torch.bool)] for i in range(len(sampling_results))]
        cls_score_ = [cls_score[(rois[:,0]==i).type(torch.bool)] for i in range(len(sampling_results))]
        bbox_pred_ = [bbox_pred[(rois[:,0]==i).type(torch.bool)] for i in range(len(sampling_results))]
        TSD_cls_score_ = [TSD_cls_score[(rois[:,0]==i).type(torch.bool)] for i in range(len(sampling_results))]
        TSD_bbox_pred_ = [TSD_bbox_pred[(rois[:,0]==i).type(torch.bool)] for i in range(len(sampling_results))]

        cls_reg_targets = bbox_target_tsd(
            pos_proposals,
            neg_proposals,
            pos_gt_bboxes,
            pos_gt_labels,
            rois_,
            delta_c_,
            delta_r_,
            cls_score_,
            bbox_pred_,
            TSD_cls_score_,
            TSD_bbox_pred_,
            rcnn_train_cfg,
            reg_classes,
            cls_pc_margin=self.cls_pc_margin,
            loc_pc_margin=self.loc_pc_margin,
            target_means=self.target_means,
            target_stds=self.target_stds)
        return cls_reg_targets

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'TSD_cls_score', 'TSD_bbox_pred', 'pc_cls_loss', 'pc_loc_loss'))
    def loss(self,
             cls_score,
             bbox_pred,
             TSD_cls_score,
             TSD_bbox_pred,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             TSD_labels, TSD_label_weights, TSD_bbox_targets, TSD_bbox_weights, pc_cls_loss, pc_loc_loss, 
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['acc'] = accuracy(cls_score, labels)
        if TSD_cls_score is not None:
            avg_factor = max(torch.sum(TSD_label_weights > 0).float().item(), 1.)
            if TSD_cls_score.numel() > 0:
                losses['loss_TSD_cls'] = self.loss_cls(
                    TSD_cls_score,
                    TSD_labels,
                    TSD_label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['TSD_acc'] = accuracy(TSD_cls_score, TSD_labels)

        if bbox_pred is not None:
            pos_inds = labels > 0
            if pos_inds.any():
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
        if TSD_bbox_pred is not None:
            pos_inds = TSD_labels > 0
            if pos_inds.any():
                if self.reg_class_agnostic:
                    TSD_bbox_pred = TSD_bbox_pred.view(
                        TSD_bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    TSD_bbox_pred = TSD_bbox_pred.view(
                        TSD_bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           TSD_labels[pos_inds.type(torch.bool)]]
                losses['loss_TSD_bbox'] = self.loss_bbox(
                    TSD_bbox_pred,
                    TSD_bbox_targets[pos_inds.type(torch.bool)],
                    TSD_bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=TSD_bbox_targets.size(0),
                    reduction_override=reduction_override)
        if pc_cls_loss is not None:
            losses['loss_pc_cls'] = pc_cls_loss.mean()
        if pc_loc_loss is not None:
            losses['loss_pc_loc'] = pc_loc_loss.mean()

        return losses

@HEADS.register_module
class TSDSharedFCBBoxHead(TSDConvFCBBoxHead):

    def __init__(self, num_fcs=2, fc_out_channels=1024, *args, **kwargs):
        assert num_fcs >= 1
        super(TSDSharedFCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=num_fcs,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
