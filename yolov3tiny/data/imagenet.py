import torch
import torchvision

import numpy as np

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
        print("to square")
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
    print("transforming...")
    transform = torchvision.transforms.Compose(
        [
            # torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((img_size, img_size)),
            torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue),
            # ToSquare(),
            # torchvision.transforms.RandomHorizontalFlip(p=0.5),
            # torchvision.transforms.RandomVerticalFlip(p=0.5),
        ]
    )
    return transform

def prepare_for_imagenet_validation(img_size):
    transform = torchvision.transforms.Compose(
        [
            # torchvision.transforms.ToTensor(),
            # ToSquare(),
            torchvision.transforms.Resize((img_size, img_size)),
        ]
    )
    return transform

class DataPrefetcher():
    def __init__(self, dataloader):
        self.dataloader = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.input, self.target = next(self.dataloader)
        except StopIteration:
            self.input = None
            self.target = None
            return

        with torch.cuda.stream(self.stream):
            self.input = self.input.cuda(non_blocking=True)
            self.target = self.target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.input
        target = self.target

        if input is not None:
            input.record_stream(torch.cuda.current_stream())

        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()

        return input, target

def collate_imagenet(samples):
    images = [sample[0] for sample in samples]
    labels = torch.tensor([sample[1] for sample in samples])
    tensor = torch.zeros((len(images), 3, images[0].size[1], images[0].size[0]))

    for i, image in enumerate(images):
        npimg = np.asarray(image)
        npimg = np.rollaxis(npimg, 2)
        tensor[i] = torch.from_numpy(npimg)
    return tensor, labels

def build_pretraining_dataloader(path,
                                 split,
                                 img_size,
                                 batch_size,
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
        dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_imagenet, num_workers=2
    )

