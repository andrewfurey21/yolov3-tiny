import torch
import torchvision
from torchvision.datasets import CocoDetection
from typing import List, Any, Tuple

from PIL import Image

def get_names(names_from_paper:str, actual_names:str):
    with open(names_from_paper) as f:
        paper = {line.strip(): i for i, line in enumerate(f)}

    with open(actual_names) as f:
        names = [line.strip() for line in f]
        keys = {paper[name]: i for i, name in enumerate(names)}
        indices = {i: name for i, name in enumerate(names)}

    return keys, indices

def center_to_topleft(bbox: torch.Tensor):
    bbox[..., 0] -= bbox[..., 2] / 2
    bbox[..., 1] -= bbox[..., 3] / 2
    return bbox

def topleft_to_center(bbox: torch.Tensor):
    bbox[..., 0] += bbox[..., 2] / 2
    bbox[..., 1] += bbox[..., 3] / 2
    return bbox

class LabelCompose(torchvision.transforms.Compose):
    def __call__(self, image: Image.Image, label:torch.Tensor = None):
        for transform in self.transforms:
            image, label = transform(image, label)
        return image, label

class ToSquare:
    def __init__(self, fill=127):
        self.fill = fill

    def __call__(self, image: torch.Tensor, label: torch.Tensor = None):
        h, w = image.shape[-2:]
        diff = abs(w - h)
        pad1 = diff // 2
        pad2 = diff - pad1
        if w > h:
            padding = (0, pad1, 0, pad2)
            if label is not None:
                label[..., 1] += pad1
        else:
            padding = (pad1, 0, pad2, 0)
            if label is not None:
                label[..., 0] += pad1
        padded_image = torchvision.transforms.functional.pad(image, padding, self.fill) # type: ignore
        return padded_image, label

class Resize:
    def __init__(self, width:int, height:int):
        self.width = width
        self.height = height 

    def __call__(self, image: torch.Tensor, label: torch.Tensor = None):
        h, w = image.shape[-2:]
        scale_w = self.width / w
        scale_h = self.height / h
        if label is not None:
            label[..., 0] *= scale_w
            label[..., 1] *= scale_h
            label[..., 2] *= scale_w
            label[..., 3] *= scale_h
        image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(self.width, self.height), mode="bilinear")
        return image.squeeze(0), label

class ToTensor:
    def __call__(self, image: Image.Image, label: torch.Tensor = None):
        return torchvision.transforms.functional.to_tensor(image), label # type: ignore

class ColorJitter(torchvision.transforms.ColorJitter):
    def __init__(self, brightness, contrast, saturation, hue):
        super().__init__(brightness, contrast, saturation, hue)

    def __call__(self, image: torch.Tensor, label: torch.Tensor = None):
        return super().__call__(image), label

# take in image and label and output (tensor, tensor) of correct shape
# max_num_boxes?
def prepare_for_training(image_size):
    return LabelCompose(
        [
            ToTensor(),
            ColorJitter(brightness=1, contrast=1, saturation=1.5, hue=0.1),
            ToSquare(),
            Resize(image_size, image_size),
        ]
    )

def prepare_for_inference(image_size):
    return LabelCompose(
        [
            ToTensor(),
            ToSquare(),
            Resize(image_size, image_size),
        ]
    )

class CocoBoundingBoxDataset(CocoDetection):
    def __init__(self, images:str, annotations:str, category_ids:dict, image_size:int, num_classes:int, max_num_boxes:int):
        super().__init__(images, annotations)
        self.transform = prepare_for_training(image_size)
        self.num_classes = num_classes
        self.num_attributes = num_classes + 5
        self.category_ids = category_ids
        self.max_num_boxes = max_num_boxes

    def __getitem__(self, index:int): # type: ignore
        image, targets = super().__getitem__(index)
        outputs = []
        for target in targets:
            if (target['category_id'] - 1) not in self.category_ids:
                continue

            bbox = torch.tensor(target['bbox'], dtype=torch.float32, requires_grad=False)
            confidence = torch.tensor([1.0], dtype=torch.float32, requires_grad=False)
            index = self.category_ids[target['category_id'] - 1]
            label = torch.eye(self.num_classes)[index]
            output = torch.cat((bbox, confidence, label), dim=0)
            outputs.append(output)
        # TODO: images might not have objects

        if len(outputs) > 0:
            output_tensor = torch.stack(outputs)
            num_box_padding = self.max_num_boxes - output_tensor.shape[0]

            image_tensor, output_tensor = self.transform(image, output_tensor) # type: ignore
            padded_output = torch.cat((output_tensor, torch.zeros(num_box_padding, output_tensor.shape[1])), dim=0)
            return image_tensor, padded_output, output_tensor.shape[0]
        else:
            image_tensor, _ = self.transform(image, None)
            padded_output = torch.zeros(self.max_num_boxes, self.num_attributes)
            return image_tensor, padded_output, 0
