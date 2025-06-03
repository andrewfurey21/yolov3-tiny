"""
momentum=0.9
decay=0.0005

# data augmentation
saturation = 1.5
exposure = 1.5
hue=.1

max_batches = 500200
"""
import random

import wandb

CLASSES = 20
ATTRIBUTES = CLASSES + 1 + 4
IGNORE_THRESHOLD = 0.5

BATCH_SIZE = 1 # training: 32
IMAGE_SIZE = 416
MAX_TARGETS = 10

NO_OBJECT = 0.5
COORD = 5

LR = 1e-3
NUM_EPOCHS = 1

if __name__ == "__main__":
    run = wandb.init(
        entity="andrewfurey2003",
        project="yolov3-tiny training run",
        config = {
            "learning_rate": 1e-3,
            "architecture": "yolov3-tiny",
            "dataset": "Pascal VOC",
            "epochs": 10,
        },
    )

    epochs = 10
    offset = random.random() / 5
    for epoch in range(2, epochs):
        acc = 1 - 2**-epoch - random.random() / epoch - offset
        loss = 2**-epoch + random.random() / epoch + offset
        run.log({"accuracy": acc, "loss": loss})

    run.finish()

    # training_data = torchvision.datasets.VOCDetection(
    #     root="./data/voc",
    #     year="2012",
    #     image_set="train",
    #     download=True,
    # )

    # collate_fn = CollateVOC("./data/voc.names")
    # dataloader = DataLoader(training_data,
    #                         batch_size=BATCH_SIZE,
    #                         shuffle=True,
    #                         num_workers=1,
    #                         collate_fn=collate_fn) # type: ignore




