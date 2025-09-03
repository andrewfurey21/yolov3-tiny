import torch
import torchvision

import pathlib

def get_imagenet_names(path:str):
    """
    takes in synset mapping, outputs dict index -> name
    """
    names = {}
    with open(path, "r") as f:
        for (i, line) in enumerate(f):
            names[i] = line.split(" ")[1].strip(",\n")
    return names

class ToSquare:
    def __init__(self, fill=127):
        self.fill = fill

    def __call__(self, image: torch.Tensor):
        h, w = image.shape[-2:]
        diff = abs(w - h)
        pad1 = diff // 2
        pad2 = diff - pad1
        if w > h:
            padding = (0, pad1, 0, pad2)
        else:
            padding = (pad1, 0, pad2, 0)
        padded_image = torchvision.transforms.functional.pad(image, padding, self.fill) # type: ignore
        return padded_image

def prepare_for_imagenet_training(img_size, brightness, contrast, saturation, hue):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue),
            ToSquare(),
            torchvision.transforms.Resize((img_size, img_size)),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomVerticalFlip(p=0.5),
        ]
    )
    return transform

def prepare_for_imagenet_validation(img_size):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            ToSquare(),
            torchvision.transforms.Resize((img_size, img_size)),
        ]
    )
    return transform

def collate_imagenet(device:torch.device):
    def collate_fn(sample):
        images, labels = [], []
        for image, label in sample:
            images.append(image)
            labels.append(label)
        return torch.stack(images, dim=0).to(device), torch.tensor(labels).to(device)
    return collate_fn

def build_pretraining_dataloader(path,
                                 split,
                                 img_size,
                                 batch_size,
                                 device,
                                 brightness:float=0,
                                 contrast:float=0,
                                 saturation:float=0,
                                 hue:float=0):

    assert split == "train" or split == "val"
    transform = prepare_for_imagenet_training(img_size, brightness, contrast, saturation, hue)
    if split == "val":
        transform = prepare_for_imagenet_validation(img_size)

    dataset = torchvision.datasets.ImageFolder(root=pathlib.Path(path)/split, transform=transform)
    sampler = torch.utils.data.RandomSampler(dataset, replacement=False)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_imagenet(device)
    )

