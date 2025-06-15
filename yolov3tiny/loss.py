from typing import Tuple, List

import torch

def iou(bboxes1: torch.Tensor, bboxes2: torch.Tensor, center_aligned=False, center_to_topleft=False):
    """
        Calculate the intersection / union between boxes1 and boxes2 (per batch)

        bboxes1 shape: (batch_size, num_b1, 4)
        bboxes2 shape: (batch_size, num_b2, 4)

        center_aligned: boxes center are aligned, useful for getting iou of anchor boxes to choose best one.
        center_to_topleft: convert coords in format (cx, cy, w, h) (x,y, w, h)

        output: (batch_size, num_b1, num_b2)
    """

    x1, x2 = bboxes1[..., 0], bboxes2[..., 0]
    y1, y2 = bboxes1[..., 1], bboxes2[..., 1]
    w1, w2 = bboxes1[..., 2], bboxes2[..., 2]
    h1, h2 = bboxes1[..., 3], bboxes2[..., 3]

    area1 = (w1 * h1).unsqueeze(2)
    area2 = (w2 * h2).unsqueeze(1)

    if center_aligned:
        w = torch.minimum(w1.unsqueeze(2), w2.unsqueeze(1)).clamp(min=0)
        h = torch.minimum(h1.unsqueeze(2), h2.unsqueeze(1)).clamp(min=0)
    else:
        if center_to_topleft:
            x1 = x1 - w1 / 2
            x2 = x2 - w2 / 2

            y1 = y1 - h1 / 2
            y2 = y2 - h2 / 2

        right1 = (x1 + w1).unsqueeze(2)
        right2 = (x2 + w2).unsqueeze(1)
        left1 = x1.unsqueeze(2)
        left2 = x2.unsqueeze(1)
        w = (torch.minimum(right1, right2) - torch.maximum(left1, left2)).clamp(min=0)

        top1 = y1.unsqueeze(2)
        top2 = y2.unsqueeze(1)
        bottom1 = (y1 + h1).unsqueeze(2)
        bottom2 = (y2 + h2).unsqueeze(1)
        h = (torch.minimum(bottom1, bottom2) - torch.maximum(top1, top2)).clamp(min=0)

    intersection = w * h

    return intersection / (area1 + area2 - intersection + 1e-9) # might not need epsilon


def object_mask(pred: torch.Tensor, target: torch.Tensor):
    """
        Creates a mask, same shape as pred without box dims (batch_size, num_predictions),
        1 == max iou with target
        0 == otherwise
    """
    batch_size, num_boxes, num_attributes = pred.shape
    assert batch_size == target.shape[0] and num_attributes == target.shape[2]
    ious = iou(pred[..., :4], target[..., :4], center_to_topleft=True)
    _, max_ious_indices = torch.max(ious, dim=1)
    mask = torch.zeros(batch_size, num_boxes).scatter_(1, max_ious_indices, 1).unsqueeze_(2)
    return mask

def preprocess_targets(target_batch: torch.Tensor, num_targets: torch.Tensor,
                       list_anchors: List[Tuple[int, int]], image_size:int):
    """
        Converting the target boxes into the form the model predicts.
        x,y should be between (image_size, image_size)

        bx = sigmoid(tx) + cx, tx = logit(bx-cx). we want tx. same for y.
        bw = pw * e^tw, tw = ln(bw/pw). we want tw. same for box height
    """

    wh_anchors = torch.tensor(list_anchors).to(target_batch.device).float()
    xy_anchors = torch.zeros(wh_anchors.shape, device=target_batch.device)
    bbox_anchors = torch.cat((xy_anchors, wh_anchors), dim=1).unsqueeze(0)

    iou_anchors_target = iou(bbox_anchors, target_batch[..., :4], center_aligned=True)
    anchor_index = torch.argmax(iou_anchors_target, dim=1)

class YOLOLoss(torch.nn.Module):
    def __init__(self, anchors, image_size, ignore_thresh, no_object_coeff, coord_coeff):
        super().__init__()
        self.anchors = anchors
        self.image_size = image_size
        self.ignore_thresh = ignore_thresh
        self.no_object_coeff = no_object_coeff
        self.coord_coeff = coord_coeff

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, num_targets: torch.Tensor):
        noobj_mask = no_object_mask(predictions, targets, self.ignore_thresh)

