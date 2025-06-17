import torch
from torchvision.ops.boxes import generalized_box_iou

batched_giou = torch.vmap(generalized_box_iou)
bce_with_logits = torch.nn.functional.binary_cross_entropy_with_logits

class YOLOLoss(torch.nn.Module):
    def __init__(self,
                 coord_weight:float=5,
                 object_weight:float=2,
                 no_object_weight:float=0.5,
                 class_weight:float=1,
                 max_num_boxes:int=100):
        super().__init__()
        self.coord_weight = coord_weight
        self.object_weight = object_weight
        self.no_object_weight = no_object_weight
        self.class_weight = class_weight
        self.max_num_boxes = max_num_boxes

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, num_targets: torch.Tensor):
        batch_size, num_predictions, num_attributes = predictions.shape
        arange = torch.arange(self.max_num_boxes).unsqueeze(0)
        len_mask = (arange < num_targets.unsqueeze(-1))

        # coord loss
        gious = batched_giou(predictions[..., :4], targets[..., :4])
        max_gious, max_giou_indices = torch.max(gious, dim=1)
        coord_loss = ((1 - max_gious) * len_mask)
        reduced_coord_loss = coord_loss.sum(dim=1).mean() * self.coord_weight

        # object loss
        offset = (torch.arange(batch_size) * num_predictions).reshape(batch_size, 1)
        max_giou_indices_offset = max_giou_indices + offset

        obj_mask = torch.zeros(batch_size * num_predictions).scatter_(0, max_giou_indices_offset[len_mask], 1).reshape(batch_size, num_predictions)
        object_loss = bce_with_logits(predictions[..., 4], torch.ones(predictions.shape[:-1]), reduction="none") * obj_mask
        reduced_object_loss = object_loss.sum(dim=1).mean() * self.object_weight

        # no object loss
        # might need an ignore mask
        no_obj_mask = torch.logical_not(obj_mask)
        no_object_loss = bce_with_logits(predictions[..., 4], torch.zeros(predictions.shape[:-1]), reduction="none") * no_obj_mask
        reduced_no_object_loss = no_object_loss.sum(dim=1).mean() * self.no_object_weight

        # class loss
        all_class_probs = predictions[..., 5:].reshape(-1, num_attributes - 5)
        class_probs = all_class_probs[max_giou_indices_offset[len_mask], :]
        target_class_probs = targets[..., 5:][len_mask]
        class_loss = bce_with_logits(class_probs, target_class_probs, reduction="none")
        reduced_class_loss = class_loss.sum(dim=1).mean() * self.class_weight

        total_loss = reduced_coord_loss + reduced_object_loss + reduced_no_object_loss + reduced_class_loss
        return total_loss.mean(), reduced_coord_loss, reduced_object_loss, reduced_no_object_loss, reduced_class_loss


