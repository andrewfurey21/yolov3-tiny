import torch
from yolov3tiny.model import YOLOLayer

img_size = 608

class PrintShape(torch.nn.Module):
    def __init__(self, i:int):
        super().__init__()
        self.i = i

    def forward(self, input:torch.Tensor):
        # print(f"Layer: {self.i}, shape: {input.shape}")
        return input

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(0)
        )

        # downsample
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(1)
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(2)
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(3)
        )

        # layer4 = shortcut(-3)

        # downsample
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(5)
        )

        self.layer6 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(6)
        )

        self.layer7 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(7)
        )

        # layer8 = shortcut(-3)

        self.layer9 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(9)
        )

        self.layer10 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(10)
        )

        # layer11 = shortcut(-3)

        # downsample
        self.layer12 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(12)
        )

        self.layer13 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(13)
        )

        self.layer14 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(14)
        )

        # layer15 = shortcut(-3)

        self.layer16 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(16)
        )

        self.layer17 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(17)
        )

        # layer18 = shortcut(-3)

        self.layer19 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(19)
        )

        self.layer20 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(20)
        )

        # layer21 = shortcut(-3)

        self.layer22 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(22)
        )

        self.layer23 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(23)
        )

        # layer24 = shortcut(-3)

        self.layer25 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(25)
        )

        self.layer26 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(26)
        )

        # layer27 = shortcut(-3)

        self.layer28 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(28)
        )

        self.layer29 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(29)
        )

        # layer30 = shortcut(-3)

        self.layer31 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(31)
        )

        self.layer32 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(32)
        )

        # layer33 = shortcut(-3)

        self.layer34 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(34)
        )

        self.layer35 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(35)
        )

        # layer36 = shortcut(-3)

        # downsample
        self.layer37 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(37)
        )

        self.layer38 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(38)
        )

        self.layer39 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(39)
        )

        # layer40 = shortcut(-3)

        self.layer41 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(41)
        )

        self.layer42 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(42)
        )

        # layer43 = shortcut(-3)

        self.layer44 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(44)
        )

        self.layer45 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(45)
        )

        # layer46 = shortcut(-3)

        self.layer47 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(47)
        )

        self.layer48 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(48)
        )

        # layer49 = shortcut(-3)

        self.layer50 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(50)
        )

        self.layer51 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(51)
        )

        # layer52 = shortcut(-3)

        self.layer53 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(53)
        )

        self.layer54 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(54)
        )

        # layer55 = shortcut(-3)

        self.layer56 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(56)
        )

        self.layer57 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(57)
        )

        # layer58 = shortcut(-3)

        self.layer59 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(59)
        )

        self.layer60 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(60)
        )

        # layer61 = shortcut(-3)

        # downsample
        self.layer62 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=1024),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(62)
        )

        self.layer63 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(63)
        )

        self.layer64 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=1024),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(64)
        )

        # layer65 = shortcut(-3)

        self.layer66 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(66)
        )

        self.layer67 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=1024),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(67)
        )

        # layer68 = shortcut(-3)

        self.layer69 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(69)
        )

        self.layer70 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=1024),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(70)
        )

        # layer71 = shortcut(-3)

        self.layer72 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(72)
        )

        self.layer73 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=1024),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(73)
        )

        # layer74 = shortcut(-3)

        self.layer75 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(75)
        )

        self.layer76 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=1024),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(76)
        )

        self.layer77 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(77)
        )

        self.layer78 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=1024),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(78)
        )

        self.layer79 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(79)
        )

        self.layer80 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=1024),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(80)
        )

        self.layer81 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1024, out_channels=255, kernel_size=1, stride=1, padding=0, bias=True),
            YOLOLayer(85, [(116, 90), (156, 198), (373, 326)], img_size),
            PrintShape(81)
        )

        self.layer82 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.UpsamplingNearest2d(scale_factor=2),
            PrintShape(82)
        )

        # 83 = route -1, 61

        self.layer84 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=768, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(84)
        )

        self.layer85 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(85)
        )

        self.layer86 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(86)
        )

        self.layer87 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(87)
        )

        self.layer88 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(88)
        )

        self.layer89 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(89)
        )

        self.layer90 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512, out_channels=255, kernel_size=1, stride=1, padding=0, bias=True),
            YOLOLayer(85, [(30, 61), (62, 45), (59, 119)], img_size),
            PrintShape(90)
        )

        self.layer91 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.UpsamplingNearest2d(scale_factor=2),
            PrintShape(91)
        )

        # layer92: route -1 (91), 36

        self.layer93 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=384, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(93)
        )

        self.layer94 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(94)
        )

        self.layer95 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(95)
        )

        self.layer96 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(96)
        )

        self.layer97 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(97)
        )

        self.layer98 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(negative_slope=0.1),
            PrintShape(98)
        )

        self.layer99 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=255, kernel_size=1, stride=1, padding=0, bias=True),
            YOLOLayer(85, [(10, 13), (16, 30), (33, 23)], img_size),
            PrintShape(99)
        )

        self.sppconv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.LeakyReLU(negative_slope=0.1),
        )


    def forward(self, input:torch.Tensor):
        output0 = self.layer0(input)
        output1 = self.layer1(output0)
        output2 = self.layer2(output1)
        output3 = self.layer3(output2)
        output4 = output3 + output1
        output5 = self.layer5(output4)
        output6 = self.layer6(output5)
        output7 = self.layer7(output6)
        output8 = output7 + output5
        output9 = self.layer9(output8)
        output10 = self.layer10(output9)
        output11 = output10 + output8
        output12 = self.layer12(output11)
        output13 = self.layer13(output12)
        output14 = self.layer14(output13)
        output15 = output14 + output12
        output16 = self.layer16(output15)
        output17 = self.layer17(output16)
        output18 = output17 + output15
        output19 = self.layer19(output18)
        output20 = self.layer20(output19)
        output21 = output20 + output18
        output22 = self.layer22(output21)
        output23 = self.layer23(output22)
        output24 = output23 + output21
        output25 = self.layer25(output24)
        output26 = self.layer26(output25)
        output27 = output26 + output24
        output28 = self.layer28(output27)
        output29 = self.layer29(output28)
        output30 = output29 + output27
        output31 = self.layer31(output30)
        output32 = self.layer32(output31)
        output33 = output32 + output30
        output34 = self.layer34(output33)
        output35 = self.layer35(output34)
        output36 = output35 + output33
        output37 = self.layer37(output36)
        output38 = self.layer38(output37)
        output39 = self.layer39(output38)
        output40 = output39 + output37
        output41 = self.layer41(output40)
        output42 = self.layer42(output41)
        output43 = output42 + output40
        output44 = self.layer44(output43)
        output45 = self.layer45(output44)
        output46 = output45 + output43
        output47 = self.layer47(output46)
        output48 = self.layer48(output47)
        output49 = output48 + output46
        output50 = self.layer50(output49)
        output51 = self.layer51(output50)
        output52 = output51 + output49
        output53 = self.layer53(output52)
        output54 = self.layer54(output53)
        output55 = output54 + output52
        output56 = self.layer56(output55)
        output57 = self.layer57(output56)
        output58 = output57 + output55
        output59 = self.layer59(output58)
        output60 = self.layer60(output59)
        output61 = output60 + output58
        output62 = self.layer62(output61)
        output63 = self.layer63(output62)
        output64 = self.layer64(output63)
        output65 = output64 + output62
        output66 = self.layer66(output65)
        output67 = self.layer67(output66)
        output68 = output67 + output65
        output69 = self.layer69(output68)
        output70 = self.layer70(output69)
        output71 = output70 + output68
        output72 = self.layer72(output71)
        output73 = self.layer73(output72)
        output74 = output73 + output71

        output75 = self.layer75(output74)
        output76 = self.layer76(output75)
        output77 = self.layer77(output76)

        spp1 = torch.nn.MaxPool2d(kernel_size=5, stride=1, padding=2)(output77)
        spp2 = torch.nn.MaxPool2d(kernel_size=9, stride=1, padding=4)(output77)
        spp3 = torch.nn.MaxPool2d(kernel_size=13, stride=1, padding=6)(output77)

        output77 = self.sppconv(torch.cat([spp1, spp2, spp3, output77], dim=1))

        output78 = self.layer78(output77)
        output79 = self.layer79(output78)
        output80 = self.layer80(output79)
        output81 = self.layer81(output80) # output 1

        output82 = self.layer82(output79)
        output83 = torch.cat([output82, output61], dim=1)
        output84 = self.layer84(output83)
        output85 = self.layer85(output84)
        output86 = self.layer86(output85)
        output87 = self.layer87(output86)
        output88 = self.layer88(output87)
        output89 = self.layer89(output88)
        output90 = self.layer90(output89) # output 2

        output91 = self.layer91(output88)
        output92 = torch.cat([output91, output36], dim=1)
        output93 = self.layer93(output92)
        output94 = self.layer94(output93)
        output95 = self.layer95(output94)
        output96 = self.layer96(output95)
        output97 = self.layer97(output96)
        output98 = self.layer98(output97)
        output99 = self.layer99(output98) # output 3

        return torch.cat([output81, output90, output99], dim=1)

if __name__ == "__main__":
    model = Model()
    input = torch.randn((1, 3, img_size, img_size))
    output = model(input)
    model.eval()
    print(output.shape)

    params = sum(p.numel() for p in model.parameters())
    buffers = sum(b.numel() for b in model.buffers())
    print("Params: ", params)
    print("buffers: ", buffers)
    s = params + buffers

    print("total weights: ", s)
    expected = 63052381 # spp
    # expected = 62001757
    print("should be ", expected)
    print("diff (+72 because of track_running_stats buffer in bn): ", s - expected)
