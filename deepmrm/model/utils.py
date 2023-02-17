import math
from collections import OrderedDict
from typing import List, Tuple, Optional

import torch
from torch import Tensor, nn
from torchvision.models.detection._utils import BoxCoder
from torchvision.models.detection.image_list import ImageList
from torchvision.ops.boxes import _upcast



def encode_lines(references: Tensor, proposals: Tensor, weights: Tensor) -> Tensor:
    """
    Encode a set of proposals with respect to some
    reference boxes

    Args:
        references (Tensor): reference boundaries
        proposals (Tensor): boundaries to be encoded
        weights (Tensor[2]): the weights for ``(x, w)``
    """
    eps = 1e-12
    wx, ww = weights

    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_x2 = proposals[:, 1].unsqueeze(1)

    reference_boxes_x1 = references[:, 0].unsqueeze(1)
    reference_boxes_x2 = references[:, 1].unsqueeze(1)

    # implementation starts here
    ex_widths = proposals_x2 - proposals_x1
    ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    # to avoid divide-by-zero
    ex_widths[ex_widths==0] = eps

    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    # to avoid divide-by-zero
    gt_widths[gt_widths==0] = eps

    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    
    targets = torch.cat((targets_dx, targets_dw), dim=1)
    # targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return targets


class BoundaryCoder(BoxCoder):
    """
    This class encodes and decodes a set of boundaries into
    the representation used for training the regressors.
    """

    def __init__(self, weights=(1.0, 1.0), bbox_xform_clip=math.log(1000.0 / 16)):
        super().__init__(weights=weights, bbox_xform_clip=bbox_xform_clip)

    # def encode(self, reference_boxes: List[Tensor], proposals: List[Tensor]) -> List[Tensor]:
    #     boxes_per_image = [len(b) for b in reference_boxes]
    #     reference_boxes = torch.cat(reference_boxes, dim=0)
    #     proposals = torch.cat(proposals, dim=0)
    #     targets = self.encode_single(reference_boxes, proposals)
    #     return targets.split(boxes_per_image, 0)

    def encode_single(self, reference_boxes: Tensor, proposals: Tensor) -> Tensor:
        """
        Encode a set of proposals with respect to some
        reference boxes

        Args:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        targets = encode_lines(reference_boxes, proposals, weights)

        return targets

    def decode(self, rel_codes: Tensor, boxes: List[Tensor]) -> Tensor:
        assert isinstance(boxes, (list, tuple))
        assert isinstance(rel_codes, torch.Tensor)
        boxes_per_image = [b.size(0) for b in boxes]
        concat_boxes = torch.cat(boxes, dim=0)
        box_sum = 0
        for val in boxes_per_image:
            box_sum += val
        if box_sum > 0:
            rel_codes = rel_codes.reshape(box_sum, -1)
        pred_boxes = self.decode_single(rel_codes, concat_boxes)
        if box_sum > 0:
            pred_boxes = pred_boxes.reshape(box_sum, -1, len(self.weights))
        return pred_boxes

    def decode_single(self, rel_codes: Tensor, boxes: Tensor) -> Tensor:
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Args:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """

        boxes = boxes.to(rel_codes.dtype)

        widths = boxes[:, 1] - boxes[:, 0]
        ctr_x = boxes[:, 0] + 0.5 * widths

        wx, ww = self.weights
        
        dx = rel_codes[:, 0::2] / wx
        dw = rel_codes[:, 1::2] / ww

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        # dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        #pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        # pred_h = torch.exp(dh) * heights[:, None]

        # Distance from center to box's corner.
        # c_to_c_h = torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
        c_to_c_w = torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w

        pred_boxes1 = pred_ctr_x - c_to_c_w
        # pred_boxes2 = pred_ctr_y - c_to_c_h
        pred_boxes3 = pred_ctr_x + c_to_c_w
        # pred_boxes4 = pred_ctr_y + c_to_c_h
        # pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)
        pred_boxes = torch.stack((pred_boxes1, pred_boxes3), dim=2).flatten(1)
        return pred_boxes


class BoundaryAnchorGenerator(nn.Module):

    __annotations__ = {
        "cell_anchors": List[torch.Tensor],
    }

    def __init__(self, sizes):
        super().__init__()
        
        num_anchors_per_grid = torch.tensor([len(s) for s in sizes])
        if torch.any(num_anchors_per_grid - num_anchors_per_grid[0] > 0):
            # [TODO] do we need different number of anchors at different feature levels? 
            raise ValueError('Number of anchors per location at every level should be the same')

        self.sizes = sizes
        self.cell_anchors = [
            self.generate_anchors(size) for size in sizes
        ]

    # For every (aspect_ratios, scales) combination, output a zero-centered anchor with those values.
    def generate_anchors(self, scales, dtype=torch.float32, device=torch.device("cpu")):
        
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        ws = scales
        base_anchors = torch.stack([-ws, ws], dim=1) / 2
        return base_anchors.round()

    def set_cell_anchors(self, dtype: torch.dtype, device: torch.device):
        self.cell_anchors = [cell_anchor.to(dtype=dtype, device=device) for cell_anchor in self.cell_anchors]

    def num_anchors_per_location(self):
        return len(self.sizes[0])

    # def forward(self, image_list: ImageList, feature_maps: List[Tensor]) -> List[Tensor]:
    def forward(self, input_image_shape, feature_maps):
        
        # image_tensor_shape = image_list.tensors.shape
        if len(self.cell_anchors) != len(feature_maps):
            raise ValueError(
                "There needs to be a match between the number of "
                "feature maps passed and the number of sizes specified.")            

        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        self.set_cell_anchors(dtype, device)

        feature_map_widths = [feature_map.shape[-1] for feature_map in feature_maps]
        image_width = input_image_shape[-1]
        # strides in original image space equivalent to single stride in feature map
        strides = [
            torch.tensor(image_width // w, dtype=torch.int64, device=device) \
                for w in feature_map_widths
        ]
        
        cell_anchors = self.cell_anchors
        anchors_over_all_feature_maps = []
        
        for feature_map_width, stride_width, base_anchors in zip(feature_map_widths, strides, cell_anchors):
            # for each feature level
            # For output anchor, compute [x_center, y_center, x_center, y_center]
            shifts_x = torch.arange(0, feature_map_width, dtype=torch.int32, device=device) * stride_width
            
            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.
            shifts = torch.stack((shifts_x, shifts_x), dim=1)
            anchors_over_all_feature_maps.append(
                (shifts.view(-1, 1, 2) + base_anchors.view(1, -1, 2)).reshape(-1, 2)
            )

        anchors = [torch.cat(anchors_over_all_feature_maps) for _ in range(input_image_shape[0])]
        # anchors: List[List[torch.Tensor]] = []
        # for _ in range(len(image_list.image_sizes)):

        #     anchors_in_image = [anchors_per_feature_map for anchors_per_feature_map in anchors_over_all_feature_maps]
        #     anchors.append(anchors_in_image)
        # anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        return anchors        



def box_area(boxes: Tensor) -> Tensor:
    boxes = _upcast(boxes)
    return (boxes[:, 1] - boxes[:, 0])


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
# def _box_inter_union(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:

def line_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Return intersection-over-union (Jaccard index) between two sets of boxes.

    Both sets of boxes are expected to be in ``(x1, x2)`` format with
    ``0 <= x1 < x2``.

    Args:
        boxes1 (Tensor[N, 2]): first set of boxes
        boxes2 (Tensor[M, 2]): second set of boxes

    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    # boxes1 = torch.tensor([[10, 20]])
    # boxes2 = torch.tensor([[15, 30], [5, 15] ])

    #inter, union = _box_inter_union(boxes1, boxes2)
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    # rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    lt = torch.max(boxes1[:, None, :1], boxes2[:, :1])  # [N,M,1]
    rb = torch.min(boxes1[:, None, 1:], boxes2[:, 1:])  # [N,M,1]    

    wh = _upcast(rb - lt).clamp(min=0)  # [N,M,1]
    # inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    inter = wh.squeeze(-1)

    union = area1[:, None] + area2 - inter
    
    iou = inter / union

    return iou

