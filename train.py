import random
import wandb

"""
momentum=0.9
decay=0.0005

# data augmentation
saturation = 1.5
exposure = 1.5
hue=.1

max_batches = 500200
"""

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

