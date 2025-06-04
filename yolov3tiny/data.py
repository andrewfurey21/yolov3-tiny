import torch
import torchvision
from torchvision.datasets import CocoDetection
from typing import List, Any, Tuple

from PIL import Image

def center_to_topleft(bbox: torch.Tensor):
    bbox[..., 0] -= bbox[..., 2] / 2
    bbox[..., 1] -= bbox[..., 3] / 2
    return bbox

def topleft_to_center(bbox: torch.Tensor):
    bbox[..., 0] += bbox[..., 2] / 2
    bbox[..., 1] += bbox[..., 3] / 2
    return bbox

class LabelCompose(torchvision.transforms.Compose):
    def __call__(self, image: Image.Image, label=torch.Tensor):
        for transform in self.transforms:
            image, label = transform(image, label)
        return image, label

class ToSquare:
    def __init__(self, fill=127):
        self.fill = fill

    def __call__(self, image: torch.Tensor, label: torch.Tensor):
        h, w = image.shape[-2:]
        diff = abs(w - h)
        pad1 = diff // 2
        pad2 = diff - pad1
        if w > h:
            padding = (0, pad1, 0, pad2)
            label[..., 1] += pad1
        else:
            padding = (pad1, 0, pad2, 0)
            label[..., 0] += pad1
        padded_image = torchvision.transforms.functional.pad(image, padding, self.fill) # type: ignore
        return padded_image, label

class Resize:
    def __init__(self, width:int, height:int):
        self.width = width
        self.height = height 

    def __call__(self, image: torch.Tensor, label: torch.Tensor):
        c, h, w = image.shape
        print(image.shape)
        scale_w = self.width / w
        scale_h = self.height / h
        label[..., 0] *= scale_w
        label[..., 1] *= scale_h
        label[..., 2] *= scale_w
        label[..., 3] *= scale_h
        image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(self.width, self.height), mode="bilinear")
        return image.squeeze(0), label

class ToTensor:
    def __call__(self, image: Image.Image, label: torch.Tensor):
        return torchvision.transforms.functional.to_tensor(image), label # type: ignore

class ColorJitter(torchvision.transforms.ColorJitter):
    def __init__(self, brightness, contrast, saturation, hue):
        super().__init__(brightness, contrast, saturation, hue)

    def __call__(self, image: torch.Tensor, label: torch.Tensor):
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

class CocoBoundingBoxDataset(CocoDetection):
    def __init__(self, images:str, annotations:str, category_ids:dict, image_size:int, num_classes:int):
        super().__init__(images, annotations)
        self.image_size = image_size
        self.transform = prepare_for_training(self.image_size)
        self.num_classes = num_classes
        self.category_ids = category_ids

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
        output_tensor = torch.stack(outputs)

        image_tensor, output_tensor = self.transform(image, output_tensor) # type: ignore
        return image_tensor, output_tensor, output_tensor.shape[0]
