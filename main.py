import torchvision
from torch.utils.data import DataLoader
import torch

import numpy as np
from sklearn.cluster import KMeans

CLASSES = 20
ATTRIBUTES = CLASSES + 1 + 4

class Convolution(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, negative_slope=0.1):
        super().__init__()
        padding = 1 if kernel_size == 3 else 0 # in yolov3-tiny, kernel sizes will only be 1 or 3
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope)

    def forward(self, input):
        output = self.conv(input)
        output = self.batch_norm(output)
        return self.leaky_relu(output)

class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer_0 = Convolution(3, 16, 3)
        self.conv_layer_2 = Convolution(16, 32, 3)
        self.conv_layer_4 = Convolution(32, 64, 3)
        self.conv_layer_6 = Convolution(64, 128, 3)
        self.conv_layer_8 = Convolution(128, 256, 3)
        self.conv_layer_10 = Convolution(256, 512, 3)
        self.conv_layer_12 = Convolution(512, 1024, 3)

        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, input):
        temp_output = self.conv_layer_0(input)
        temp_output = self.maxpool(temp_output)

        temp_output = self.conv_layer_2(temp_output)
        temp_output = self.maxpool(temp_output)

        temp_output = self.conv_layer_4(temp_output)
        temp_output = self.maxpool(temp_output)

        temp_output = self.conv_layer_6(temp_output)
        temp_output = self.maxpool(temp_output)

        output_1 = self.conv_layer_8(temp_output)
        temp_output = self.maxpool(output_1)

        temp_output = self.conv_layer_10(temp_output)

        # pad right and bottom since max pool will have a stride of 1
        temp_output = torch.nn.ConstantPad2d((0, 1, 0, 1), float('-inf'))(temp_output)
        temp_output = torch.nn.MaxPool2d(kernel_size=2, stride=1)(temp_output)

        output_2= self.conv_layer_12(temp_output)
        return output_1, output_2

class YOLOLayer(torch.nn.Module):
    def __init__(self, anchors):
        super().__init__()
        self.anchors = anchors

    def forward(self, input):
        batch_size = input.shape[0]
        grid_size = input.shape[2]

        new_view = input.view(batch_size, 3, ATTRIBUTES, grid_size, grid_size)
        permutation = new_view.permute(0, 1, 3, 4, 2)
        contiguous = permutation.contiguous()
        return contiguous.view(batch_size, -1, ATTRIBUTES)

class YOLOv3tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = Backbone()

        self.conv_layer_13 = Convolution(1024, 256, 1)
        self.conv_layer_14 = Convolution(256, 512, 3)
        self.conv_layer_15 = Convolution(512, 3 * ATTRIBUTES, 1)

        self.conv_layer_18 = Convolution(256, 128, 1)
        self.conv_layer_21 = Convolution(384, 256, 3)
        self.conv_layer_22 = Convolution(256, 3 * ATTRIBUTES, 1)

        self.yolo_layer_larger = YOLOLayer([(81,82),  (135,169),  (344,319)]) # TODO: use kmeans to calculate from dataset
        self.yolo_layer_smaller = YOLOLayer([(10,14),  (23,27),  (37,58)])


    def forward(self, input):
        layer_8, layer_12 = self.backbone(input)

        output_layer_13 = self.conv_layer_13(layer_12)
        temp_output = self.conv_layer_14(output_layer_13)
        temp_output = self.conv_layer_15(temp_output)
        yolo_output_1 = self.yolo_layer_larger(temp_output)

        temp_output = self.conv_layer_18(output_layer_13)
        temp_output = torch.nn.UpsamplingNearest2d(scale_factor=2)(temp_output)
        temp_output = torch.cat([temp_output, layer_8], dim=1)
        temp_output = self.conv_layer_21(temp_output)
        temp_output = self.conv_layer_22(temp_output)
        yolo_output_2 = self.yolo_layer_smaller(temp_output)

        final_output = torch.cat([yolo_output_1, yolo_output_2], dim=1)
        return final_output

# Function for calculating the loss function:
# 1. iou batch
# 2. pre process targets
# 3. no object mask function
# 4. no object mask filter
# 5. yolo loss function

if __name__ == "__main__":
    batch_size = 1

    input = torch.arange(0, batch_size*416*416*3).type(torch.float32).reshape(batch_size, 3, 416, 416)
    model = YOLOv3tiny()
    output = model(input)

    print("Input shape: ", input.shape)
    print("Output shape: ", output.shape)

