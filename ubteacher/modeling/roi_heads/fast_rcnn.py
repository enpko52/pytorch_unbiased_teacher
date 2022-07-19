# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import cat
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers


class FastRCNNFocaltLossOutputLayers(FastRCNNOutputLayers):
    def __init__(self, cfg, input_shape):
        super(FastRCNNFocaltLossOutputLayers, self).__init__(cfg, input_shape)
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES

    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features
                that were used to compute predictions.
        """
        scores, proposal_deltas = predictions

        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        loss_cls = self.focal_loss(scores, gt_classes)

        losses = {
            "loss_cls": loss_cls,
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }

        return losses

    def focal_loss(self, pred_class_logits, gt_classes):
        """ Compute the focal loss """

        if pred_class_logits.numel() == 0:
            return 0.0 * self.pred_class_logits.sum()
        else:
            fc_loss = FocalLoss(
                gamma=1.5,
                num_classes=self.num_classes,
            )

            total_loss = fc_loss(pred_class_logits, gt_classes)
            total_loss = total_loss / gt_classes.shape[0]

            return total_loss


class FocalLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        gamma=1.0,
        num_classes=80,
    ):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

        self.num_classes = num_classes

    def forward(self, input, target):
        # focal loss
        CE = F.cross_entropy(input, target, reduction="none")
        p = torch.exp(-CE)
        loss = (1 - p) ** self.gamma * CE
        return loss.sum()
