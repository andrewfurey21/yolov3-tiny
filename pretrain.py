import torch
import torchvision
import wandb
from yolov3tiny import data, model, draw

import os
from datetime import datetime
os.makedirs(name="./checkpoints", exist_ok=True)

def display_image_tensor(image: torch.Tensor, labels:torch.Tensor, size:int, num_classes:int):
    assert size <= labels.shape[1]
    names_from_paper = "./data/coco-paper.names"
    actual_names = "./data/coco.names"
    _, indices = data.get_names(names_from_paper, actual_names)
    class_ids = torch.argmax(labels[:size, 5:], dim=1).tolist()
    class_names = [indices[id] for id in class_ids]
    pilimage = torchvision.transforms.functional.to_pil_image(image) # type: ignore

    draw.draw_bboxes(pilimage,
                     labels[:size, :4].tolist(),
                     class_names,
                     class_ids,
                     num_classes)
    pilimage.show() # type: ignore

# TODO: make directory with unique name in checkpoints so that different training runs can be saved.
def checkpoint(epoch:int, model:model.YOLOv3tinyPretrain, optimizer:torch.optim.Optimizer, loss:float):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "loss": loss,
    }, f"./checkpoints/epoch_{epoch}_loss_{loss:.4f}.pt")

if __name__ == "__main__":
    torch.manual_seed(12345)

    # optimizer
    lr = 0.001
    adamw_betas = (0.9, 0.999)
    weight_decay = 0.01

    # hyperparams
    epochs = 10
    batch_size = 2
    img_size = 416
    num_classes = 1000

    # lr schedule (StepLR)
    step_size = 400_000
    gamma = 0.1

    # data augmentation
    saturation = 1.5
    exposure = 1.5
    hue = 0.1


    #device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model
    pretrain_model = model.YOLOv3tinyPretrain(num_classes).to(device)

    # optimizer
    optim = torch.optim.AdamW(pretrain_model.parameters(), lr=lr, betas=adamw_betas, weight_decay=weight_decay) # set fused and capturable

    # loss + learning rate scheduling
    lossfn = torch.nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=step_size, gamma=gamma)

    # logging
    save_interval = 1000
    run_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    pretraining_log = wandb.init(
        entity="andrewfurey2003",
        project="yolov3-tiny-pytorch",
        name=run_name,
        group="pretraining-imagenet",
        config= {
            "learning_rate": lr,
            "architecture": "yolov3-tiny",
            "dataset": "ImageNet",
        },
        mode="online",
    )

    # example
    for i in range(100):
        input = torch.rand((2, 3, 224, 224))
        output = pretrain_model(input)

        actual = torch.rand((2, 1000))
        loss = lossfn(output, actual)

        loss.backward()
        optim.step()
        lr_scheduler.step()

        print(f"Epoch {i+1}: Loss={loss}, LR={lr_scheduler.get_last_lr()}")

        if i % save_interval == 0:
            checkpoint(i, pretrain_model, optim, loss)
        pretraining_log.log({"Loss": loss})

    pretraining_log.finish()
