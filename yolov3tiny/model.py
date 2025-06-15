import torch
from typing import List, Tuple

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
    def __init__(self, num_attributes:int, anchors:List[Tuple[int, int]], img_size:int):
        super().__init__()
        self.num_attributes = num_attributes
        assert len(anchors) == 3
        self.anchors = torch.tensor(anchors)
        self.img_size = img_size

    def forward(self, input):
        batch_size, channels, grid_size, _grid_size = input.shape
        assert self.img_size % grid_size == 0
        assert channels == self.num_attributes * 3
        stride = self.img_size // grid_size

        input = input.reshape(batch_size, 3, self.num_attributes, grid_size, grid_size) # (n,c,w,h) -> (n, 3, c/3, w, h)
        input = input.permute(0, 1, 3, 4, 2) # (n, 3, w, h, c/3)

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

        # objectness score
        conf = torch.sigmoid(input[..., 4]).unsqueeze(4)

        # class confidence scores
        class_  = torch.softmax(input[..., 5:], dim=4)

        return torch.cat([bbox, conf, class_], dim=4).reshape(batch_size, -1, self.num_attributes)

class YOLOv3tiny(torch.nn.Module):
    def __init__(self, num_classes:int, anchors:List[Tuple[int, int]], img_size:int):
        super().__init__()
        self.num_attributes = num_classes + 5
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # layers numbered based off yolov3-tiny architecture diagram.
        self.conv_layer_0 = Convolution(3, 16, 3)
        self.conv_layer_2 = Convolution(16, 32, 3)
        self.conv_layer_4 = Convolution(32, 64, 3)
        self.conv_layer_6 = Convolution(64, 128, 3)
        self.conv_layer_8 = Convolution(128, 256, 3)
        self.conv_layer_10 = Convolution(256, 512, 3)
        self.conv_layer_12 = Convolution(512, 1024, 3)

        self.conv_layer_13 = Convolution(1024, 256, 1)

        self.conv_layer_14 = Convolution(256, 512, 3)
        self.conv_layer_15 = Convolution(512, 3 * self.num_attributes, 1)

        self.conv_layer_18 = Convolution(256, 128, 1)
        self.conv_layer_21 = Convolution(384, 256, 3)
        self.conv_layer_22 = Convolution(256, 3 * self.num_attributes, 1)

        self.yolo_layer = YOLOLayer(self.num_attributes, anchors[3:], img_size)
        self.yolo_layer_upsampled = YOLOLayer(self.num_attributes, anchors[:3], img_size)


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

        pad_a10 = torch.nn.ConstantPad2d((0, 1, 0, 1), float('-inf'))(a10)
        a11 = torch.nn.MaxPool2d(kernel_size=2, stride=1)(pad_a10)
        a12 = self.conv_layer_12(a11)
        a13 = self.conv_layer_13(a12)

        a14 = self.conv_layer_14(a13)
        a15 = self.conv_layer_15(a14)
        output_1 = self.yolo_layer(a15)

        a18 = self.conv_layer_18(a13)
        a19 = torch.nn.UpsamplingNearest2d(scale_factor=2)(a18)
        a20 = torch.cat([a19, a8], dim=1)
        a21 = self.conv_layer_21(a20)
        a22 = self.conv_layer_22(a21)
        output_2 = self.yolo_layer_upsampled(a22)

        return torch.cat([output_1, output_2], dim=1)
