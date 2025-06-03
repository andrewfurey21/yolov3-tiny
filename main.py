import torchvision
from torch.utils.data import DataLoader, Subset
import torch
import torch.nn.functional as F

from typing import List, Tuple, Any
import cv2
import colorsys

import constants

class CollateVOC():
    names_file: str
    keys: dict

    def __init__(self, names_file:str):
        self.names_file = names_file
        with open(self.names_file, 'r') as file:
            self.keys = {line.strip(): i for i, line in enumerate(file)}

    def __call__(self, sample:Tuple[Any, Any]) -> Any:

        """
            each "batch" here is a tuple, consiting of a PIL.Image and a dictionary,
            representing the XML tree structure of the bounding box info
            and other meta data

            what we want:
            1. images: tensor with a certain batch size, not PIL.Image shape=(batch_size, 3, width, height)
            2. bounding_boxes: tensor, shape=(batch_size, number of boxes for this image, ATTRIBUTES)
        """

        # TODO: needs a redo, loads of loops
        # pretty sure pascal VOC uses [xmin, ymin, xmax, ymax]. need to convert if not done by pytorch
        images = torch.stack([torchvision.transforms.ToTensor()(image) for image, _ in sample], dim=0)
        bounding_boxes = []
        for image in sample:
            for _object in image[1]["annotation"]["object"]:
                labels = [float(value) for value in _object["bndbox"].values()]
                labels += [1.0 if i == self.keys[_object["name"]] else 0.0 for i in range(CLASSES)]
                bounding_boxes.append(labels)
        return images, torch.tensor(bounding_boxes) #, lengths

def iou(bboxes1: torch.Tensor, bboxes2: torch.Tensor, center_aligned=False, center_format=False):
    """ 
        Calculate the intersection / union between boxes1 and boxes2 (per batch)

        bboxes1, bboxes2 shape: (batch_size, number_of_boxes, 4)

        center_aligned: boxes center are aligned, useful for getting iou of anchor boxes to choose best one.
        center_format: are boxes coords in format (cx, cy, w, h) or x,y in top left.

        output: (batch_size, n1, n2)
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
        if center_format:
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


def no_object_mask(pred: torch.Tensor, target: torch.Tensor):
    """
        Creates a mask, same shape as pred without box dims (batch_size, num_predictions),
        1 == no prediction with a good iou in target
        0 otherwise
    """
    batch_size, _, num_attributes = pred.shape
    assert batch_size == target.shape[0], num_attributes == target.shape[2]
    ious = iou(pred[..., :4], target[..., :4], center_format=True)
    return torch.max(ious, dim=2)[0] < IGNORE_THRESHOLD

def no_object_mask_filter(noobj_mask: torch.Tensor, object_index: torch.Tensor):
    """
        noobj_mask: mask, 1 == no prediction with decent iou in targets
        object_index: indices where theres an object (in target)
    """
    batch_size, num_predictions = noobj_mask.shape
    noobj_mask = noobj_mask.view(-1) # flatten
    _filter = torch.zeros(noobj_mask.shape, device=noobj_mask.device)
    noobj_mask.scatter_(0, object_index, _filter)
    return noobj_mask.view(batch_size, num_predictions)

# TODO: should work for variable sized image_size so that people can train smaller models faster
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

    scale = anchor_index // 3 # 0 or 1
    cell_size = image_size / 26 # 416 / 26 = 16
    stride = (cell_size * 2 ** scale).float() # 16 for upsampled (smaller boxes), 32 otherwise

    cx = target_batch[..., 0] // stride # which grid cell?
    cy = target_batch[..., 1] // stride

    tx = (target_batch[..., 0] / stride - cx).clamp(1e-9, 1-1e-9) # need to clamp so logit function works
    tx = torch.log(tx / (1 - tx))

    ty = (target_batch[..., 1] / stride - cy).clamp(1e-9, 1-1e-9)
    ty = torch.log(ty / (1 - ty))

    chosen_anchors = torch.index_select(wh_anchors, 0, torch.flatten(anchor_index)).reshape(tuple(list(anchor_index.shape) + [2])) # make this better
    tw = torch.log(target_batch[..., 2] / chosen_anchors[..., 0])
    th = torch.log(target_batch[..., 3] / chosen_anchors[..., 1])

    target_batch_t = target_batch.clone().detach()

    target_batch_t[..., 0] = tx
    target_batch_t[..., 1] = ty
    target_batch_t[..., 2] = tw
    target_batch_t[..., 3] = th

    large_scale_mask = scale.logical_not().long()
    grid_size = image_size // stride

    object_index = (large_scale_mask * (image_size // 32) ** 2 * 3 \
            + grid_size ** 2 * (anchor_index % 3) \
            + grid_size * cy \
            + cx).long()

    object_index_flat = []
    targets_flat = []

    batch_size = target_batch.shape[0]
    num_predictions = 13 ** 2 * 3 + 26 ** 2 * 3

    for batch in range(batch_size):
        indices = object_index[batch]
        targets = target_batch_t[batch]
        length = num_targets[batch]
        object_index_flat.append(indices[:length] + batch * num_predictions)
        targets_flat.append(targets[:length])

    return torch.cat(targets_flat), torch.cat(object_index_flat)

class YOLOLoss(torch.nn.Module):
    def __init__(self, anchors, image_size):
        super().__init__()
        self.anchors = anchors
        self.image_size = image_size

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, num_targets: torch.Tensor):
        targets, target_indices = preprocess_targets(targets, num_targets, self.anchors, self.image_size)
        noobj_mask = no_object_mask(predictions, targets)
        noobj_mask = no_object_mask_filter(noobj_mask, indices)

        # Loss
        # 1. no object loss: supress false positives
        # 2. object loss: suppress false negatives
        # 3. class loss
        # 4. coord loss
        # 5. loss = no object loss + object loss + class loss + coord loss

        # no objectness loss
        confidence_logits = predictions[..., 4]
        target_confidence_noobj = torch.zeros_like(confidence_logits)
        noobj_confidence_logits = confidence_logits - (1 - noobj_mask) * 1e7
        noobj_loss = F.binary_cross_entropy_with_logits(noobj_confidence_logits, target_confidence_noobj, reduction="sum")

        batch_size, num_predictions, _ = predictions.shape
        preds_obj = predictions.view(batch_size * num_predictions, -1).index_select(0, target_indices)

        coord_loss = F.mse_loss(preds_obj[..., :4], targets[..., :4], reduction="sum")

        target_confidence_obj = torch.ones_like(preds_obj[..., 4])
        obj_loss = F.binary_cross_entropy_with_logits(preds_obj[..., 4], target_confidence_noobj, reduction="sum")

        class_loss = F.binary_cross_entropy_with_logits(preds_obj[..., 5:], targets[..., 5:], reduction="sum")

        total_loss = class_loss + obj_loss + COORD * coord_loss + NO_OBJECT * noobj_loss

        return total_loss, coord_loss, obj_loss, noobj_loss, class_loss


def rescale_bbox_to_image():
    pass

def box_colour(class_id:int, num_classes:int) -> Tuple:
    h = float(class_id) / float(num_classes)
    r, g, b = colorsys.hsv_to_rgb(h, 1, 1)
    return (int(r * 255), int(g * 255), int(b * 255))[::-1]

def draw_bboxes(image, boxes:List[Tuple], class_ids: List[int], class_names:List[str]):
    """
        image:
        boxes: List[Tuple[x1, y1, x2, y2]]. 
        positions have to be scaled correctly to original image size (rescale_bbox_to_image)
        class_names: List[str]
    """
    assert len(boxes) == len(class_ids)

    for box, class_id in zip(boxes, class_ids):
        title = class_names[class_id]
        colour = box_colour(class_id, len(class_names))
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), colour, 4)
        cv2.putText(image, title, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)

def inference_single_image(model, input_file_name, output_file_name, class_names):
    image = cv2.imread(input_file_name)

    positions = [(50, 50, 100, 100),
               (75, 90, 180, 200),
               (300, 40, 350, 145)]
    classes = [1, 0, 1]

    draw_bboxes(image, positions, classes, class_names)
    cv2.imwrite(output_file_name, image)

if __name__ == "__main__":
    # training_data = torchvision.datasets.VOCDetection(
    #     root="./data/voc",
    #     year="2012",
    #     image_set="train",
    #     download=True,
    # )

    # collate_fn = CollateVOC("./data/voc.names")
    # dataloader = DataLoader(training_data,
    #                         batch_size=BATCH_SIZE,
    #                         shuffle=True,
    #                         num_workers=1,
    #                         collate_fn=collate_fn) # type: ignore

    model = YOLOv3tiny()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    input = torch.randn((1, 3, 416, 416))
    output= model(input)
    print(output.shape)



