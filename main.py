import torchvision
from torch.utils.data import DataLoader, Subset
import torch

from typing import Tuple, Any
# import numpy as np
# from sklearn.cluster import KMeans

CLASSES = 20 # number of classes in pascal VOC dataset
ATTRIBUTES = CLASSES + 1 + 4 # number of classes + objectness score + x,y,w,h

class Convolution(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, negative_slope=0.1):
        assert kernel_size == 1 or kernel_size == 3
        super().__init__()
        padding = 1 if kernel_size == 3 else 0
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope)

    def forward(self, input):
        output = self.conv(input)
        output = self.batch_norm(output)
        return self.leaky_relu(output)

class YOLOLayer(torch.nn.Module):
    def __init__(self, anchors):
        super().__init__()
        self.anchors = anchors

    def forward(self, input):
        # why not adjusting the anchor boxes?
        # TODO: draw diagram of whats going on
        batch_size = input.shape[0]
        grid_size = input.shape[2]

        new_view = input.view(batch_size, 3, ATTRIBUTES, grid_size, grid_size)
        permutation = new_view.permute(0, 1, 3, 4, 2)
        contiguous = permutation.contiguous()
        return contiguous.view(batch_size, -1, ATTRIBUTES)

class YOLOv3tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # numbered based off yolov3-tiny architecture diagram.
        self.conv_layer_0 = Convolution(3, 16, 3)
        self.conv_layer_2 = Convolution(16, 32, 3)
        self.conv_layer_4 = Convolution(32, 64, 3)
        self.conv_layer_6 = Convolution(64, 128, 3)
        self.conv_layer_8 = Convolution(128, 256, 3)
        self.conv_layer_10 = Convolution(256, 512, 3)
        self.conv_layer_12 = Convolution(512, 1024, 3)

        self.conv_layer_13 = Convolution(1024, 256, 1)

        self.conv_layer_14 = Convolution(256, 512, 3)
        self.conv_layer_15 = Convolution(512, 3 * ATTRIBUTES, 1)

        self.conv_layer_18 = Convolution(256, 128, 1)
        self.conv_layer_21 = Convolution(384, 256, 3)
        self.conv_layer_22 = Convolution(256, 3 * ATTRIBUTES, 1)

        # TODO: use kmeans to calculate from dataset
        self.yolo_layer_larger = YOLOLayer([(81,82),  (135,169),  (344,319)])
        self.yolo_layer_smaller = YOLOLayer([(10,14),  (23,27),  (37,58)])


    def forward(self, input):
        a0 = self.conv_layer_0(input)
        a1 = self.maxpool(a0)
        a2 = self.conv_layer_2(a1)
        a3 = self.maxpool(a2)
        a4 = self.conv_layer_4(a3)
        a5 = self.maxpool(a4)
        a6 = self.conv_layer_6(a5)
        a7 = self.maxpool(a6)
        a8 = self.conv_layer_8(a7)

        a9 = self.maxpool(a8)
        a10 = self.conv_layer_10(a9)
        # pad right and bottom since max pool will have a stride of 1
        pad_a10 = torch.nn.ConstantPad2d((0, 1, 0, 1), float('-inf'))(a10)
        a11 = torch.nn.MaxPool2d(kernel_size=2, stride=1)(pad_a10)
        a12 = self.conv_layer_12(a11)
        a13 = self.conv_layer_13(a12)

        a14 = self.conv_layer_14(a13)
        a15 = self.conv_layer_15(a14)
        output_1 = self.yolo_layer_larger(a15)

        a18 = self.conv_layer_18(a13)
        a19 = torch.nn.UpsamplingNearest2d(scale_factor=2)(a18)
        a20 = torch.cat([a19, a8], dim=1)
        a21 = self.conv_layer_21(a20)
        a22 = self.conv_layer_22(a21)
        output_2 = self.yolo_layer_smaller(a22)

        final_output = torch.cat([output_1, output_2], dim=1)
        return final_output

# Function for calculating the loss function:
# 1. iou batch
# 2. pre process targets
# 3. no object mask function
# 4. no object mask filter
# 5. yolo loss function

def iou(bboxes1: torch.Tensor, bboxes2: torch.Tensor, center_aligned=False, center_format=False):
    """ 
        Calculate the intersection / union between boxes1 and 
        boxes2 (per batch)

        bboxes1, bboxes2 shape: (batch_size, number_of_boxes, 4)

        center_aligned: boxes center are aligned
        center_format: are boxes coords in format (cx, cy, w, h) or x,y in top left.
    """

    x1, x2 = bboxes1[..., 0], bboxes2[..., 0]
    y1, y2 = bboxes1[..., 1], bboxes2[..., 1]
    w1, w2 = bboxes1[..., 2], bboxes2[..., 2]
    h1, h2 = bboxes1[..., 3], bboxes2[..., 3]

    area1 = (w1 * h1).unsqueeze(2)
    area2 = w2 * h2.unsqueeze(1)

    if center_aligned:
        w1.unsqueeze_(2)
        w2.unsqueeze_(1)
        h1.unsqueeze_(2)
        h2.unsqueeze_(1)
        w = torch.minimum(w1, w2).clamp(min=0)
        h = torch.minimum(h1, h2).clamp(min=0)
    else:
        if center_format:
            x1 = x1 - w1 / 2
            x2 = x2 - w2 / 2

            y1 = y1 - h1 / 2
            y2 = y2 - h2 / 2

        right1 = (x1 + w1).unsqueeze_(2)
        right2 = (x2 + w2).unsqueeze_(1)
        left1 = x1.unsqueeze(2)
        left2 = x2.unsqueeze(1)
        w = (torch.minimum(right1, right2) - torch.maximum(left1, left2)).clamp(min=0)

        top1 = y1.unsqueeze(2)
        top2 = y2.unsqueeze(1)
        bottom1 = (y1 + h1).unsqueeze(2)
        bottom2 = (y2 + h2).unsqueeze(1)
        h = (torch.minimum(bottom1, bottom2) - torch.maximum(top1, top2)).clamp(min=0)

    intersection = w * h
    return intersection / (area1 + area2 - intersection + 1e-9)

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

        # TODO: needs a redo, loads of loops, not sure if this is the right format
        images = torch.stack([torchvision.transforms.ToTensor()(image) for image, _ in sample], dim=0)
        bounding_boxes = []
        for image in sample:
            for _object in image[1]["annotation"]["object"]:
                labels = [float(value) for value in _object["bndbox"].values()]
                labels += [1.0 if i == self.keys[_object["name"]] else 0.0 for i in range(CLASSES)]
                bounding_boxes.append(labels)
        return images, torch.tensor(bounding_boxes)

if __name__ == "__main__":
    batch_size = 1
    # input = torch.arange(0, batch_size*416*416*3).type(torch.float32).reshape(batch_size, 3, 416, 416)
    # model = YOLOv3tiny()
    # output = model(input)
    #
    # print("Input shape: ", input.shape)
    # print("Output shape: ", output.shape)

    # train = torchvision.datasets.VOCDetection(
    #     root="./data/voc",
    #     year="2012",
    #     image_set="train",
    #     download=True,
    # )
    #
    # # TODO: transforms (need to transform box as well)
    # train_subset = Subset(train, [0])
    # # train_subset[0][0].show()
    # 
    # collate_fn = CollateVOC("./data/voc.names")
    # dataloader = DataLoader(train_subset, 
    #                         batch_size=batch_size, 
    #                         shuffle=True, 
    #                         num_workers=1, 
    #                         collate_fn=collate_fn) # type: ignore
    #
    # for image, label in dataloader:
    #     print("Image shape: ", image.shape)
    #     print(label)


    a = torch.arange(16).reshape(2, 2, 4)
    b = torch.arange(32).reshape(2, 4, 4)
    c = iou(a, b, center_aligned=True)
    print(c)




