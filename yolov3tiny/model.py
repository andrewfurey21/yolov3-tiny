import torch
from typing import List, Tuple

from yolov3tiny.data import cxcywh_to_xyxy

class Convolution(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, negative_slope=0.1, dropoutp=0.2):
        assert kernel_size == 1 or kernel_size == 3
        super().__init__()
        padding = 1 if kernel_size == 3 else 0
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope)
        self.dropout = torch.nn.Dropout2d(p=dropoutp) # Pretty sure original wasn't trained with dropout

    def forward(self, input):
        output = self.conv(input)
        output = self.batch_norm(output)
        output = self.leaky_relu(output)
        output = self.dropout(output)
        return output

class PaddingMaxPool(torch.nn.Module):
    def __init__(self, padding:Tuple, fill:float, kernel_size:int, stride:int):
        super().__init__()
        self.pad = torch.nn.ConstantPad2d(padding, fill)
        self.maxpool = torch.nn.MaxPool2d(kernel_size, stride=stride)

    def forward(self, input:torch.Tensor):
        output = self.pad(input)
        output = self.maxpool(output)
        return output

class YOLOLayer(torch.nn.Module):
    def __init__(self, num_attributes:int, anchors:List[Tuple[int, int]], img_size:int):
        super().__init__()
        self.num_attributes = num_attributes
        assert len(anchors) == 3
        self.anchors = torch.tensor(anchors)
        self.img_size = img_size

    def forward(self, input):
        batch_size, channels, width, height = input.shape
        assert width == height
        grid_size = width

        assert self.img_size % grid_size == 0 # TODO: is this really necessary?
        stride = self.img_size // grid_size

        assert channels == self.num_attributes * 3

        input = input.reshape(batch_size, 3, self.num_attributes, grid_size, grid_size) # (n,c,w,h) -> (n, 3, c/3, w, h)
        input = input.permute(0, 1, 3, 4, 2) # (n, 3, w, h, c/3), c/3 == self.num_attributes

        offset = torch.arange(grid_size).repeat(3, grid_size, 1) # (3, grid_size, grid_size)
        x_offset = offset.unsqueeze(0) # (1, 3, grid_size, grid_size)
        y_offset = offset.transpose(-1, -2).unsqueeze(0)

        anchor_w = self.anchors[..., 0].reshape(3, 1, 1)
        anchor_h = self.anchors[..., 1].reshape(3, 1, 1)

        # format: cx, cy, w, h
        x_pred = (torch.sigmoid(input[..., 0]) + x_offset) * stride
        y_pred = (torch.sigmoid(input[..., 1]) + y_offset) * stride

        w_pred = anchor_w * torch.exp(input[..., 2])
        h_pred = anchor_h * torch.exp(input[..., 3])

        bbox = torch.stack([x_pred, y_pred, w_pred, h_pred], dim=4)
        bbox = cxcywh_to_xyxy(bbox)

        probs = torch.sigmoid(input[..., 4:])
        return torch.cat([bbox, probs], dim=4).reshape(batch_size, -1, self.num_attributes)

class YOLOv3tiny(torch.nn.Module):
    def __init__(self, num_classes:int, anchors:List[Tuple[int, int]], img_size:int):
        super().__init__()
        self.num_attributes = num_classes + 5
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer_0 = Convolution(3, 16, 3)
        self.conv_layer_2 = Convolution(16, 32, 3)
        self.conv_layer_4 = Convolution(32, 64, 3)
        self.conv_layer_6 = Convolution(64, 128, 3)
        self.conv_layer_8 = Convolution(128, 256, 3)
        self.conv_layer_10 = Convolution(256, 512, 3)

        self.maxpool_11 = PaddingMaxPool((0, 1, 0, 1), float('-inf'), 2, 1)
        self.conv_layer_12 = Convolution(512, 1024, 3)

        self.conv_layer_13 = Convolution(1024, 256, 1)

        self.conv_layer_14 = Convolution(256, 512, 3)
        self.conv_layer_15 = Convolution(512, 3 * self.num_attributes, 1)

        self.conv_layer_17 = Convolution(256, 128, 1)
        self.upsample_18 = torch.nn.UpsamplingNearest2d(scale_factor=2)
        self.conv_layer_19 = Convolution(384, 256, 3)
        self.conv_layer_20 = Convolution(256, 3 * self.num_attributes, 1)

        self.yolo_layer_16 = YOLOLayer(self.num_attributes, anchors[3:], img_size)
        self.yolo_layer_21 = YOLOLayer(self.num_attributes, anchors[:3], img_size)


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

        a11 = self.maxpool_11(a10)
        a12 = self.conv_layer_12(a11)
        a13 = self.conv_layer_13(a12)

        a14 = self.conv_layer_14(a13)
        a15 = self.conv_layer_15(a14)
        a16 = self.yolo_layer_16(a15)

        a17 = self.conv_layer_17(a13)
        a18 = self.upsample_18(a17)
        a18a8 = torch.cat([a18, a8], dim=1)
        a19 = self.conv_layer_19(a18a8)
        a20 = self.conv_layer_20(a19)
        a21 = self.yolo_layer_21(a20)

        return torch.cat([a16, a21], dim=1)

class YOLOv3tinyPretrain(torch.nn.Module):
    def __init__(self, num_classes:int):
        super().__init__()
        self.num_attributes = num_classes
        self.file_name = f"{__class__.__name__}.pt"
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer_0 = Convolution(3, 16, 3)
        self.conv_layer_2 = Convolution(16, 32, 3)
        self.conv_layer_4 = Convolution(32, 64, 3)
        self.conv_layer_6 = Convolution(64, 128, 3)
        self.conv_layer_8 = Convolution(128, 256, 3)
        self.conv_layer_10 = Convolution(256, 512, 3)

        self.maxpool_11 = PaddingMaxPool((0, 1, 0, 1), float('-inf'), 2, 1)
        self.conv_layer_12 = Convolution(512, 1024, 3)

        self.conv_layer_13 = Convolution(1024, 256, 1)

        self.conv_layer_17 = Convolution(256, 128, 1)
        self.upsample_18 = torch.nn.UpsamplingNearest2d(scale_factor=2)
        self.conv_layer_19 = Convolution(384, 256, 3)

        self.global_average_pool_20 = torch.nn.AvgPool2d(kernel_size=14)
        self.dense_21 = torch.nn.Linear(256, num_classes, bias=True)

    def forward(self, input:torch.Tensor):
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

        a11 = self.maxpool_11(a10)
        a12 = self.conv_layer_12(a11)
        a13 = self.conv_layer_13(a12)

        a17 = self.conv_layer_17(a13)
        a18 = self.upsample_18(a17)
        a18a8 = torch.cat([a18, a8], dim=1)
        a19 = self.conv_layer_19(a18a8)

        # imagenet classification
        a20 = self.global_average_pool_20(a19).flatten(start_dim=1)
        a21 = self.dense_21(a20)

        return a21
