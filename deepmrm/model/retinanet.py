from collections import OrderedDict
from typing import Dict, List, Tuple, Optional
import math

import torch
from torch import nn, Tensor

from torchvision.ops import boxes as box_ops
from torchvision.utils import _log_api_usage_once

from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.retinanet import (
    RetinaNetClassificationHead, RetinaNetRegressionHead,
    _sum
)
from deepmrm.model.utils import BoundaryCoder, line_iou


class ClassificationHead(RetinaNetClassificationHead):

    def __init__(self, in_channels, num_anchors, num_classes, prior_probability=0.01, num_conv_layers=1):
        super().__init__(in_channels, num_anchors, num_classes, prior_probability)
        conv = []
        for _ in range(num_conv_layers):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), stride=1, padding=(0, 1)))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

        self.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=(1, 3), stride=1, padding=(0, 1))
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        torch.nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability))

    def forward(self, x):
        # type: (List[Tensor]) -> Tensor
        all_cls_logits = []
        for features in x:
            cls_logits = self.conv(features)
            cls_logits = self.cls_logits(cls_logits)
            
            # Permute classification output from (N, A * K, H, W) to (N, HWA, K).
            N, _, H, W = cls_logits.shape
            cls_logits = cls_logits.view(N, -1, self.num_classes, H, W)
            cls_logits = cls_logits.permute(0, 3, 4, 1, 2)
            cls_logits = cls_logits.reshape(N, -1, self.num_classes)  # Size=(N, HWA, 4)

            all_cls_logits.append(cls_logits)

        return torch.cat(all_cls_logits, dim=1)

    
class RegressionHead(RetinaNetRegressionHead):
    """
    A regression head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
    """

    def __init__(self, in_channels, num_anchors, num_conv_layers=1):
        super().__init__(in_channels, num_anchors)
        
        num_output_points = 2 # it's 1D boundary detector

        # collapse along the height axis
        conv = []
        for _ in range(num_conv_layers):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), stride=1, padding=(0, 1)))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        self.bbox_reg = nn.Conv2d(in_channels, num_anchors*num_output_points, kernel_size=(1, 3), stride=1, padding=(0, 1))
        torch.nn.init.normal_(self.bbox_reg.weight, std=0.01)
        torch.nn.init.zeros_(self.bbox_reg.bias)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.zeros_(layer.bias)

        self.num_output_points = num_output_points
        self.box_coder = BoundaryCoder()

    def forward(self, x):
        # type: (List[Tensor]) -> Tensor
        all_bbox_regression = []
        P = self.num_output_points

        for features in x:
            # features = torch.rand([32, 256, 1, 59])
            bbox_regression = self.conv(features)
            bbox_regression = self.bbox_reg(bbox_regression)

            # Permute bbox regression output from (N, P * A, H, W) to (N, HWA, P).
            N, _, H, W = bbox_regression.shape
            bbox_regression = bbox_regression.view(N, -1, P, H, W)
            bbox_regression = bbox_regression.permute(0, 3, 4, 1, 2)
            bbox_regression = bbox_regression.reshape(N, -1, P)  # Size=(N, HWA, P)

            all_bbox_regression.append(bbox_regression)

        return torch.cat(all_bbox_regression, dim=1)

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor], List[Tensor]) -> Tensor
        losses = []
        bbox_regression = head_outputs["bbox_regression"]

        for targets_per_image, bbox_regression_per_image, anchors_per_image, matched_idxs_per_image in zip(
            targets, bbox_regression, anchors, matched_idxs
        ):
            # determine only the foreground indices, ignore the rest
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            num_foreground = foreground_idxs_per_image.numel()

            # select only the foreground boxes
            matched_gt_boxes_per_image = targets_per_image["boxes"][matched_idxs_per_image[foreground_idxs_per_image]]
            matched_gt_labels_per_image = targets_per_image["labels"][matched_idxs_per_image[foreground_idxs_per_image]]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]

            # compute the regression targets
            target_regression = self.box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)

            #### weighted loss
            # l1_loss = torch.nn.functional.l1_loss(bbox_regression_per_image, target_regression, reduction='none')
            # weighted_l1_sum = 0.5*torch.dot(l1_loss.sum(axis=1), matched_gt_labels_per_image.type(torch.float32))
            # loss = weighted_l1_sum/max(1, num_foreground)

            #### non-weighted loss
            loss = torch.nn.functional.l1_loss(bbox_regression_per_image, target_regression, reduction="sum") / max(1, num_foreground)

            # compute the loss
            losses.append(loss)

        return _sum(losses) / max(1, len(targets))        


class RetinaNetHead(nn.Module):
    
    def __init__(self, in_channels, num_anchors, num_classes, num_conv_layers=1):
        super().__init__()
        self.classification_head = ClassificationHead(in_channels, num_anchors, num_classes, num_conv_layers=num_conv_layers)
        self.regression_head = RegressionHead(in_channels, num_anchors, num_conv_layers=num_conv_layers)

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor], List[Tensor]) -> Dict[str, Tensor]
        return {
            "classification": self.classification_head.compute_loss(targets, head_outputs, matched_idxs),
            "bbox_regression": self.regression_head.compute_loss(targets, head_outputs, anchors, matched_idxs),
        }

    def forward(self, x):
        # type: (List[Tensor]) -> Dict[str, Tensor]
        return {"cls_logits": self.classification_head(x), "bbox_regression": self.regression_head(x)}


class RetinaNet(nn.Module):

    def __init__(
        self,
        backbone,
        anchor_generator,
        num_classes=3,
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=10,
        fg_iou_thresh=0.5,
        bg_iou_thresh=0.4,
        num_conv_layers_in_head=1,
        # topk_candidates=1000,
    ):
        super().__init__()
        _log_api_usage_once(self)

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )
        
        # 1. backbone
        self.backbone = backbone
        
        # 2. anchor_generator
        self.anchor_generator = anchor_generator

        # 3. head
        num_anchors = self.anchor_generator.num_anchors_per_location()
        self.head = RetinaNetHead(backbone.out_channels, num_anchors, num_classes, num_conv_layers_in_head)
        
        # 4. proposal matcher
        self.proposal_matcher = det_utils.Matcher(
                fg_iou_thresh,
                bg_iou_thresh,
                allow_low_quality_matches=True,)

        # 5. box coder
        self.box_coder = BoundaryCoder()

        # 6. a few thresholds
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        # self.topk_candidates = topk_candidates

    def forward(self, input_tensors, targets=None):
        if targets:
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 2:
                        raise ValueError(f"Expected target boundaries to be a tensor of shape [N, 2], got {boxes.shape}.")
                else:
                    raise ValueError(f"Expected target boundaries to be of type Tensor, got {type(boxes)}.")

        # get the features from the backbone
        features = self.backbone(input_tensors)
        features = list(features.values())

        # compute the retinanet heads outputs using the features
        head_outputs = self.head(features)
        # { 'cls_logits': [#samples, #anchors, #classes],
        #   'bbox_regression': [#samples, #anchors, 2] }
        
        # create the list of anchors, the size of each is [#anchors, 2]
        anchors = self.anchor_generator(input_tensors.shape, features)

        return features, head_outputs, anchors

    def compute_loss(self, xic_tensors, targets):
        features, head_outputs, anchors = self(xic_tensors, targets)
        matched_idxs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            if targets_per_image["boxes"].numel() == 0:
                matched_idxs.append(
                    torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device)
                )
                continue
            
            # [#samples, #anchors]
            match_quality_matrix = line_iou(targets_per_image["boxes"], anchors_per_image)
            matched_idxs.append(self.proposal_matcher(match_quality_matrix))

        return self.head.compute_loss(targets, head_outputs, anchors, matched_idxs)
  

    def predict(self, xic_tensors):
        features, head_outputs, anchors = self(xic_tensors)

        # features: #feature_maps x [#samples, C, W, H]
        # head_outputs: {'class_logits': [#samples, #anchors, #classes], .... }
        # anchors: #samples X [#anchors, 2]
        num_anchors_per_level = [
                x.size(3)*self.anchor_generator.num_anchors_per_location() for x in features
            ]

        split_head_outputs = {
            k: list(head_outputs[k].split(num_anchors_per_level, dim=1)) for k in head_outputs
        }
        # split_head_outputs: {'cls_logits': #feature_maps x [#samples, #anchors_in_level, #classes], ...}
        
        # split_anchors: #samples x #feature_maps x [#anchors_in_level, 2]
        split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

        # head_outputs = split_head_outputs
        # anchors = split_anchors
        # image_shape = xic_tensors.shape[-2:]
        # compute the detections
        detections = self.postprocess_detections(split_head_outputs, split_anchors, xic_tensors.shape[-2:])

        return detections

    def postprocess_detections(self, head_outputs, anchors, image_shape):
        # type: (Dict[str, List[Tensor]], List[List[Tensor]], List[Tuple[int, int]]) -> List[Dict[str, Tensor]]
        
        # head_outputs: {'cls_logits': #feature_maps x [#samples, #anchors_in_level, #classes], ...}
        num_samples = len(anchors)
        class_logits = head_outputs["cls_logits"]
        box_regression = head_outputs["bbox_regression"]

        detections: List[Dict[str, Tensor]] = []
        for index in range(num_samples):
            
            # #feature_maps x [#anchors_in_level, #classes]
            logits_per_image = [cl[index] for cl in class_logits]
            box_regression_per_image = [br[index] for br in box_regression]
            
            anchors_per_image, image_shape = anchors[index], image_shape

            image_boxes = []
            image_scores = []
            image_labels = []
            # image_all_scores = []
            
            # process each feature-map level
            for box_regression_per_level, logits_per_level, anchors_per_level in zip(
                box_regression_per_image, logits_per_image, anchors_per_image
            ):
                # box_regression_per_level.shape
                # logits_per_level.shape
                # anchors_per_level.shape

                # [NOTE] only one anchor per each level! (no aspect ratio, no height)
                assert anchors_per_level.dim() == 2
                num_classes = logits_per_level.shape[-1]
                
                scores = torch.sigmoid(logits_per_level)
                # # multi-label vs multi-class task
                # if hasattr(self, 'is_multilabel') and self.is_multilabel:
                #     scores = torch.sigmoid(logits_per_level)
                # else:
                #     scores = torch.softmax(logits_per_level, 1)

                
                scores_per_level = scores.flatten()
                # remove low scoring boxes
                keep_idxs = scores_per_level > self.score_thresh

                scores_per_level = scores_per_level[keep_idxs]
                topk_idxs = torch.where(keep_idxs)[0]

                # keep only topk scoring predictions
                num_topk = min(1000, topk_idxs.size(0))
                scores_per_level, idxs = scores_per_level.topk(num_topk)
                topk_idxs = topk_idxs[idxs]

                anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
                labels_per_level = topk_idxs % num_classes
                
                boxes_per_level = self.box_coder.decode_single(
                    box_regression_per_level[anchor_idxs], anchors_per_level[anchor_idxs]
                )
                
                # boxes_per_level = box_ops.clip_boxes_to_image(boxes_per_level, image_shape)
                boxes_per_level = boxes_per_level.clamp(min=0, max=image_shape[1])
                image_boxes.append(boxes_per_level)
                image_scores.append(scores_per_level)
                # image_all_scores.append(scores[anchor_idxs])
                image_labels.append(labels_per_level)

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)
            # image_all_scores = torch.cat(image_all_scores, dim=0)
            image_boxes_temp = torch.zeros([image_boxes.size(0), 4], 
                                        dtype=image_boxes.dtype, 
                                        device=image_boxes.device)
            # upper-left
            image_boxes_temp[:, 0] = image_boxes[:, 0]
            image_boxes_temp[:, 1] = 0
            # lower-right
            image_boxes_temp[:, 2] = image_boxes[:, 1]
            image_boxes_temp[:, 3] = 1 # image_shape[0]

            m = (image_boxes[:, 0] - image_boxes[:, 1]).abs() > 1
            image_boxes, image_boxes_temp, image_scores, image_labels = \
                image_boxes[m], image_boxes_temp[m], image_scores[m], image_labels[m]

            # non-maximum suppression
            keep = box_ops.batched_nms(image_boxes_temp, image_scores, image_labels, self.nms_thresh)
            # tmp_image_labels = torch.ones_like(image_labels)
            # keep = box_ops.batched_nms(image_boxes_temp, image_scores, tmp_image_labels, self.nms_thresh)
            keep = keep[:self.detections_per_img]

            detections.append(
                {
                    "boxes": image_boxes[keep],
                    "scores": image_scores[keep],
                    "labels": image_labels[keep],
                }
            )

        return detections