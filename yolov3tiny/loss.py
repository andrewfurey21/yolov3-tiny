import torch
from torchvision.ops.boxes import generalized_box_iou

batched_giou = torch.vmap(generalized_box_iou)

class YOLOLoss(torch.nn.Module):
    def __init__(self, ignore_thresh:float=0.5, no_object_coeff:float=0.5, coord_coeff:float=5, max_num_boxes:int=100):
        super().__init__()
        self.ignore_thresh = ignore_thresh
        self.no_object_coeff = no_object_coeff
        self.coord_coeff = coord_coeff
        self.max_num_boxes = max_num_boxes

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, num_targets: torch.Tensor):
        arange = torch.arange(self.max_num_boxes).unsqueeze(0)
        len_mask = (arange < num_targets.unsqueeze(-1))

        # coord loss
        gious = batched_giou(predictions[..., :4], targets[..., :4])
        max_gious, max_giou_indices = torch.max(gious, dim=1)
        coord_loss = ((1 - max_gious) * len_mask).sum(dim=1)

        # object loss
        print(max_gious)

        objects = torch.gather(predictions, 1, max_giou_indices)
        # obj_mask = torch.zeros(predictions.shape[:-1]).scatter_(1, max_giou_indices, 1)
        # object_loss = torch.nn.functional.binary_cross_entropy(predictions[..., 4], torch.ones(predictions.shape[:-1]), reduction="none") * obj_mask
        # print(obj_mask)
        # print(object_loss)

        # no_obj_mask = torch.logical_not(obj_mask)
        # no object loss
        no_object_loss = 0

        # class loss
        class_loss = 0

        # total_loss = self.coord_coeff * coord_loss + object_loss + self.no_object_coeff * no_object_loss + class_loss
        # return total_loss.mean(), coord_loss, object_loss, no_object_loss, class_loss
        return 0


