import torch
import torchvision
from torchvision.datasets import CocoDetection
from typing import List, Any, Tuple

class ToSquare:
    def __init__(self, fill=127):
        self.fill = fill

    def __call__(self, image: torch.Tensor):
        w, h = image.shape[-2:]
        diff = abs(w - h)
        pad1 = diff // 2
        pad2 = diff - pad1
        padding = (0, pad1, 0, pad2) if w < h else (pad1, 0, pad2, 0)
        padded_image = torchvision.transforms.functional.pad(image, padding, self.fill) # type: ignore
        return padded_image

def center_to_topleft(bbox: torch.Tensor):
    bbox[..., 0] -= bbox[..., 2] / 2
    bbox[..., 1] -= bbox[..., 3] / 2
    return bbox

def topleft_to_center(bbox: torch.Tensor):
    bbox[..., 0] += bbox[..., 2] / 2
    bbox[..., 1] += bbox[..., 3] / 2
    return bbox

# take in image and label and output (tensor, tensor) of correct shape
# max_num_boxes?
def prepare_batch_for_training(image_size):
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.ColorJitter(brightness=1, contrast=1, saturation=1.5, hue=0.1),
            ToSquare(),
            torchvision.transforms.Resize((image_size, image_size)),
        ]
    )

class CocoBoundingBoxDataset(CocoDetection):
    def __init__(self, images:str, annotations:str, category_ids:dict, image_size:int, num_classes:int):
        super().__init__(images, annotations)
        self.image_size = image_size
        self.transform = prepare_batch_for_training(self.image_size)
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
        return image, output_tensor, output_tensor.shape[0]
