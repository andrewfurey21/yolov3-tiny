import torch
import torch.profiler
import torchvision
from torchvision.datasets import imagenet

from yolov3tiny import model, data

import os
from datetime import datetime
from dotenv import load_dotenv

import wandb
from tqdm import tqdm

CHECKPOINTS = "checkpoints" # weight checkpoints folder

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
        get the latest checkpoint. if best_loss, get the checkpoint with best loss, else get the latest one (most recent epoch)
        should be used to checkpoint weights once every epoch.
    """
    index = 3 if best_loss else 1
    files = {float(entry.name.split("_")[index]):entry.name for entry in os.scandir(f"./{CHECKPOINTS}/")}
    return files[reversed(sorted([float(entry.name.split("_")[index]) for entry in os.scandir(f"./{CHECKPOINTS}/")])).__next__()]

if __name__ == "__main__":
    # torch.manual_seed(12345)
    load_dotenv()
    os.makedirs(name=CHECKPOINTS, exist_ok=True)

    # optimizer
    lr = 0.0001
    momentum = 0.9
    weight_decay = 0.00001
    nesterov = True

    # hyperparams
    epochs = 30
    batch_size = 32
    img_size = 224
    num_classes = 1000

    # lr schedule (StepLR)
    step_size = 10
    gamma = 0.2

    # data augmentation
    brightness = 0.5
    contrast = 0.5
    saturation = 0.5
    hue = 0.5

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset and dataloader
    names_file = os.getenv("IMAGENET_NAMES")
    assert names_file, "Need to specify where the imagenet names are stored."
    imagenet_filepath = os.getenv("IMAGENET")
    assert imagenet_filepath, "Need to specify where imagenet images are."

    names = data.get_imagenet_names(names_file)
    train_dataloader = data.build_pretraining_dataloader(imagenet_filepath, "train",
                                                         img_size, batch_size,
                                                         brightness, contrast, saturation, hue)
    print("Created training dataloader.")

    # device and model
    pretrain_model = model.YOLOv3tinyPretrain(num_classes).to(device)
    pretrain_model.train()

    # optimizer + lr scheduler
    optim = torch.optim.SGD(pretrain_model.parameters(),
                            lr=lr,
                            weight_decay=weight_decay,
                            momentum=momentum,
                            nesterov=nesterov) # TODO: set fused and capturable
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=step_size, gamma=gamma)

    # loss
    lossfn = torch.nn.CrossEntropyLoss()

    if os.getenv("LOAD") != None:
        best_loss = True if int(os.getenv("LOAD")) == 1 else False # type: ignore
        checkpoint_file_name = get_latest_checkpoint(best_loss=best_loss)
        load = f"Using weights from checkpoint: {checkpoint_file_name}"
        pretrain_model.load_state_dict(torch.load(f"./{CHECKPOINTS}/{checkpoint_file_name}")["model"])


    steps = 30
    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule = torch.profiler.schedule(
            wait=1, warmup=1, active=steps, repeat=1
        ),
        profile_memory=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
        record_shapes=True,
        with_stack=True
    )

    # prefetcher = data.DataPrefetcher(train_dataloader)
    step = 0
    for (xbatch, ybatch) in train_dataloader:
    # for (xbatch, ybatch) in tqdm(train_dataloader, desc=f"Training on epoch {0}"):
    # for (step, (xbatch, ybatch)) in tqdm(enumerate((train_dataloader)), desc="training..."):
        x = xbatch.to(device, non_blocking=True)
        y = ybatch.to(device, non_blocking=True)
        ypred = pretrain_model(x)
        loss = lossfn(ypred, y)
        loss.backward()
        optim.step()
        lr_scheduler.step()

        profiler.step()

        print(step)
        if step > steps:
            profiler.stop()
            break
        else:
            step += 1

    print(profiler.key_averages().table(sort_by="self_cpu_time_total"))
