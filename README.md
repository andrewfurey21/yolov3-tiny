# yolov3-tiny in pytorch

yolov3-tiny training and inference code using pytorch.

.
├── src
│   ├── loss.py
│   ├── model.py
│   └── data.py
├── infer.py
└── train.py

* `loss.py`: loss function
* `model.py`: description of the yolov3-tiny model
* `data.py`: dealing data loaders, transforms and data augmentation
* `infer.py`: inference code, drawing bounding boxes
* `train.py`: training script, contains hyperparameters

## useful links

* This implementation is based off of [yolov3-tiny in pytorch](https://github.com/ValentinFigue/TinyYOLOv3-PyTorch)
* [YOLO paper](https://arxiv.org/pdf/1506.02640)
* [YOLOv2 paper](https://arxiv.org/pdf/1612.08242)
* [YOLOv3 paper](https://arxiv.org/pdf/1804.02767)
* [yolov3-tiny config from darknet](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg)
