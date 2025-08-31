import torch
import torchvision

from yolov3tiny import model, data

import os
from datetime import datetime
from dotenv import load_dotenv

import wandb
from tqdm import tqdm, trange

CHECKPOINTS = "checkpoints"

def display_image_tensor(image: torch.Tensor):
    pilimage = torchvision.transforms.functional.to_pil_image(image) # type: ignore
    pilimage.show()

def save_checkpoint(id:str, epoch:int, model:model.YOLOv3tinyPretrain, optimizer:torch.optim.Optimizer, loss:float):
    os.makedirs(name=f"./{CHECKPOINTS}/{id}", exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "loss": loss,
    }, f"./{CHECKPOINTS}/{id}/epoch_{epoch}_loss_{loss:.4f}_.pt")

def get_latest_checkpoint(best_loss=False):
    """
        get the latest checkpoint. if best_loss, get the checkpoint with best loss, else get the latest one (largest epoch)
    """
    index = 3 if best_loss else 1
    files = {float(entry.name.split("_")[index]):entry.name for entry in os.scandir(f"./{CHECKPOINTS}/")}
    return files[reversed(sorted([float(entry.name.split("_")[index]) for entry in os.scandir(f"./{CHECKPOINTS}/")])).__next__()]

if __name__ == "__main__":
    torch.manual_seed(12345)
    load_dotenv()
    os.makedirs(name=CHECKPOINTS, exist_ok=True)

    # optimizer
    lr = 0.0001
    momentum = 0.9
    weight_decay = 0.00001
    nesterov = True

    # hyperparams
    epochs = 200
    batch_size = 32
    img_size = 224
    num_classes = 1000

    # lr schedule (StepLR)
    step_size = 10000
    gamma = 0.9

    # data augmentation
    brightness = 0.5
    contrast = 0.5
    saturation = 0.5
    hue = 0.5

    # dataset and dataloader
    imagenet_dir = os.getenv("IMAGENET")
    assert imagenet_dir, "Must set imagenet root directory"
    dataloader = data.build_imagenet_dataloader(imagenet_dir, img_size, batch_size, brightness, contrast, saturation, hue)

    # device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrain_model = model.YOLOv3tinyPretrain(num_classes).to(device)
    pretrain_model.train()

    # optimizer + lr scheduler
    optim = torch.optim.SGD(pretrain_model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov) # TODO: set fused and capturable
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=step_size, gamma=gamma)

    # loss
    lossfn = torch.nn.CrossEntropyLoss()

    # logging
    save_interval = 1000
    run_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    pretraining_log = wandb.init(
        entity=os.getenv("ENTITY"),
        project=os.getenv("PROJECT"),
        name=run_name,
        config={
            "epochs":epochs,
            "batch size":batch_size,
            "learning rate scheduler":lr_scheduler.__class__.__name__,
            "optimizer":optim.__class__.__name__,
            "device":device.__str__()
        },
        group="pretraining-imagenet",
        mode="online",
    )

    if os.getenv("LOAD") != None:
        best_loss = True if int(os.getenv("LOAD")) == 1 else False # type: ignore
        checkpoint_file_name = get_latest_checkpoint(best_loss=best_loss)
        load = f"Using weights from checkpoint: {checkpoint_file_name}"
        # pretraining_log.notes += load
        # pretraining_log.save();

        pretrain_model.load_state_dict(torch.load(f"./{CHECKPOINTS}/{checkpoint_file_name}")["model"])

    # example
    xbatch, ybatch = next(iter(dataloader))
    # for epoch in [1]:
    for epoch in trange(epochs):
        # for (xbatch, ybatch) in tqdm(dataloader, desc=f"Training on epoch {epoch}"):
        ypred = pretrain_model(xbatch)
        loss = lossfn(ypred, ybatch)
        loss.backward()
        optim.step()
        lr_scheduler.step()
        pretraining_log.log({f"batch_loss_{0}": loss.item()})

        # calls torch.save. once every epoch, save weights.
        # save_checkpoint(wandb.run.id, i, pretrain_model, optim, loss) # type: ignore
        # validation, torch.no_grad
        # pretraining_log.log({"val_loss": loss})
        # top1 and top5
        top1 = (ypred.argmax(dim=1) == ybatch).sum() / batch_size
        top5 = (torch.topk(ypred, k=5, dim=1)[1] == ybatch.reshape(batch_size, 1)).max(dim=1)[0].float().sum() / batch_size
        pretraining_log.log({f"top 1": top1})
        pretraining_log.log({f"top 5": top5})

    pretraining_log.finish()
