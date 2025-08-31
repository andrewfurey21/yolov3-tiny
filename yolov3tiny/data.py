import torch
import torchvision
from torchvision.transforms.functional import to_tensor
from torchvision.datasets import CocoDetection

from PIL import Image

def cxcywh_to_xyxy(bbox: torch.Tensor):
    bbox[..., 0] -= bbox[..., 2] / 2
    bbox[..., 1] -= bbox[..., 3] / 2
    bbox[..., 2] = bbox[..., 0] + bbox[..., 2]
    bbox[..., 3] = bbox[..., 1] + bbox[..., 3]
    return bbox

def xywh_to_xyxy(bbox: torch.Tensor):
    bbox[..., 2] = bbox[..., 0] + bbox[..., 2]
    bbox[..., 3] = bbox[..., 1] + bbox[..., 3]
    return bbox

def get_names(names_from_paper:str, actual_names:str):
    """
    coco paper release 91 names, but the dataset only contains 80.

    keys: {name index in coco-paper : name index in actual}
    indices { index of actual: str of actual}
    """
    with open(names_from_paper) as f:
        paper = {line.strip(): i for i, line in enumerate(f)}

    with open(actual_names) as f:
        names = [line.strip() for line in f]
        keys = {paper[name]: i for i, name in enumerate(names)}
        indices = {i: name for i, name in enumerate(names)}

    return keys, indices

def prepare_for_pretraining(img_size, brightness, contrast, saturation, hue):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue),
            torchvision.transforms.Resize((img_size, img_size)),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomVerticalFlip(p=0.5),
        ]
    )
    return transform

def prepare_for_validation(img_size):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((img_size, img_size)),
        ]
    )
    return transform

class LabelCompose(torchvision.transforms.Compose):
    def __call__(self, image: Image.Image, label:torch.Tensor|None = None):
        for transform in self.transforms:
            image, label = transform(image, label)
        return image, label

class LabelToSquare:
    def __init__(self, fill=127):
        self.fill = fill

    def __call__(self, image: torch.Tensor, label: torch.Tensor|None = None):
        h, w = image.shape[-2:]
        diff = abs(w - h)
        pad1 = diff // 2
        pad2 = diff - pad1
        if w > h:
            padding = (0, pad1, 0, pad2)
            if label is not None:
                label[..., 1] += pad1
                label[..., 3] += pad1
        else:
            padding = (pad1, 0, pad2, 0)
            if label is not None:
                label[..., 0] += pad1
                label[..., 2] += pad1
        padded_image = torchvision.transforms.functional.pad(image, padding, self.fill) # type: ignore
        return padded_image, label

class LabelResize:
    def __init__(self, width:int, height:int):
        self.width = width
        self.height = height 

    def __call__(self, image: torch.Tensor, label: torch.Tensor|None = None):
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

class LabelToTensor:
    def __call__(self, image: Image.Image, label: torch.Tensor|None = None):
        return torchvision.transforms.functional.to_tensor(image), label # type: ignore

class LabelColorJitter(torchvision.transforms.ColorJitter):
    def __init__(self, brightness, contrast, saturation, hue):
        super().__init__(brightness, contrast, saturation, hue)

    def __call__(self, image: torch.Tensor, label: torch.Tensor|None = None):
        return super().__call__(image), label

class LabelRandomHorizontalFlip(torchvision.transforms.RandomHorizontalFlip):
    def __init__(self, img_size:int, p:float=0.5):
        super().__init__(p)
        self.p = p
        self.img_size = img_size

    def __call__(self, image: torch.Tensor, label: torch.Tensor|None = None):
        flipped_image = image
        if torch.rand(1) < self.p:
            flipped_image = torchvision.transforms.functional.hflip(image) # type: ignore
            if label != None:
                bbox_w = label[..., 2] - label[..., 0]
                label[..., 0] = self.img_size - label[..., 2]
                label[..., 2] = label[..., 0] + bbox_w
        return flipped_image, label

class LabelRandomVerticalFlip(torchvision.transforms.RandomVerticalFlip):
    def __init__(self, img_size:int, p:float=0.5):
        super().__init__(p)
        self.p = p
        self.img_size = img_size

    def __call__(self, image: torch.Tensor, label: torch.Tensor|None = None):
        flipped_image = image
        if torch.rand(1) < self.p:
            flipped_image = torchvision.transforms.functional.vflip(image) # type: ignore
            if label != None:
                bbox_h = label[..., 3] - label[..., 1]
                label[..., 1] = self.img_size - label[..., 3]
                label[..., 3] = label[..., 1] + bbox_h
        return flipped_image, label

def prepare_for_training(img_size:int):
    return LabelCompose(
        [
            LabelToTensor(),
            LabelColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            LabelToSquare(),
            LabelResize(img_size, img_size),
            LabelRandomHorizontalFlip(img_size, p=0.5),
            LabelRandomVerticalFlip(img_size, p=0.5),
        ]
    )

def prepare_for_inference(img_size:int):
    return LabelCompose(
        [
            LabelToTensor(),
            LabelToSquare(),
            LabelResize(img_size, img_size),
        ]
    )

class CocoBoundingBoxDataset(CocoDetection):
    def __init__(self, images:str, annotations:str, category_ids:dict, img_size:int, num_classes:int, max_num_boxes:int, transform=prepare_for_inference):
        super().__init__(images, annotations)
        self.transform = transform(img_size)
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
            fixed_bbox = xywh_to_xyxy(bbox)
            confidence = torch.tensor([1.0], dtype=torch.float32, requires_grad=False)
            index = self.category_ids[target['category_id'] - 1]
            label = torch.eye(self.num_classes)[index]
            output = torch.cat((fixed_bbox, confidence, label), dim=0)
            outputs.append(output)

        if len(outputs) > 0:
            output_tensor = torch.stack(outputs)
            num_box_padding = self.max_num_boxes - output_tensor.shape[0]

            image_tensor, output_tensor = self.transform(image, output_tensor) # type: ignore
            padded_output = torch.cat((output_tensor, torch.zeros(num_box_padding, output_tensor.shape[1])), dim=0) # type: ignore
            return image_tensor, padded_output, output_tensor.shape[0] # type: ignore
        else:
            image_tensor, _ = self.transform(image, None) # type: ignore
            padded_output = torch.zeros(self.max_num_boxes, self.num_attributes)
            return image_tensor, padded_output, 0


def build_coco_dataloader(images_dir:str, annotations_dir:str, img_size:int, num_classes:int, max_num_boxes:int, batch_size:int, replacement:bool, transform):
    names_from_paper = "./data/coco-paper.names"
    actual_names = "./data/coco.names"
    keys, _ = get_names(names_from_paper, actual_names)
    dataset = CocoBoundingBoxDataset(
        images=images_dir,
        annotations=annotations_dir,
        category_ids=keys,
        img_size=img_size,
        num_classes=num_classes,
        max_num_boxes=max_num_boxes,
        transform=transform
    )

    def collate_coco_sample(sample):
        images, labels, sizes = [], [], []
        for image, label, size in sample:
            images.append(image)
            labels.append(label)
            sizes.append(size)
        return torch.stack(images), torch.stack(labels), torch.tensor(sizes)

    # dataloading
    sampler = torch.utils.data.RandomSampler(dataset, replacement=replacement)
    dataloader = torch.utils.data.DataLoader(dataset,
                            batch_size=batch_size,
                            sampler=sampler,
                            collate_fn=collate_coco_sample)
    return dataloader

def build_imagenet_dataloader(imagenet_dir:str, img_size:int, batch_size:int, brightness:float, contrast:float, saturation:float, hue:float):
    def collage_imagenet_sample(sample):
        images, labels = [], []
        for image, label in sample:
            images.append(image)
            labels.append(label)
        return torch.stack(images, dim=0), torch.tensor(labels)
    imagenet_dataset = torchvision.datasets.ImageNet(imagenet_dir, split="val", transform=prepare_for_pretraining(img_size, brightness, contrast, saturation, hue))
    sampler = torch.utils.data.RandomSampler(imagenet_dataset, replacement=False)
    return torch.utils.data.DataLoader(
        imagenet_dataset, batch_size=batch_size, sampler=sampler, collate_fn=collage_imagenet_sample
    )
