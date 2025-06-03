import torch
import torchvision
from typing import Any, Tuple

class CollateVOC():
    names_file: str
    keys: dict

    def __init__(self, names_file:str, num_classes:int):
        self.names_file = names_file
        self.num_classes = num_classes
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
                labels += [1.0 if i == self.keys[_object["name"]] else 0.0 for i in range(self.num_classes)]
                bounding_boxes.append(labels)
        return images, torch.tensor(bounding_boxes) #, lengths
