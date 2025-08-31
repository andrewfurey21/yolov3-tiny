import torch
import torchvision
import wandb
from yolov3tiny import model, data

import os
from datetime import datetime
from dotenv import load_dotenv

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
    lr = 0.001
    adamw_betas = (0.9, 0.999)
    weight_decay = 0.01

    # hyperparams
    epochs = 2 
    batch_size = 1
    img_size = 224
    num_classes = 1000

    # lr schedule (StepLR)
    step_size = 400_000
    gamma = 0.1

    # data augmentation
    saturation = 1.5
    exposure = 1.5
    hue = 0.1

    # dataset and dataloader
    imagenet_dir = os.getenv("IMAGENET")
    assert imagenet_dir, "Must set imagenet root directory"
    dataloader = data.build_imagenet_dataloader(imagenet_dir, img_size, batch_size)

    # device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrain_model = model.YOLOv3tinyPretrain(num_classes).to(device)
    pretrain_model.train()

    # optimizer
    optim = torch.optim.AdamW(pretrain_model.parameters(), lr=lr, betas=adamw_betas, weight_decay=weight_decay) # TODO: set fused and capturable

    # loss + learning rate scheduling
    lossfn = torch.nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=step_size, gamma=gamma)

    # logging
    save_interval = 1000
    run_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    # pretraining_log = wandb.init(
    #     entity=os.getenv("ENTITY"),
    #     project=os.getenv("PROJECT"),
    #     name=run_name,
    #     group="pretraining-imagenet",
    #     mode="online",
    # )

    if os.getenv("LOAD") != None:
        best_loss = True if int(os.getenv("LOAD")) == 1 else False # type: ignore
        checkpoint_file_name = get_latest_checkpoint(best_loss=best_loss)
        load = f"Using weights from checkpoint: {checkpoint_file_name}"
        # pretraining_log.notes = load
        # pretraining_log.save();

        pretrain_model.load_state_dict(torch.load(f"./{CHECKPOINTS}/{checkpoint_file_name}")["model"])

    # example
    for epoch in range(epochs):
        for xbatch, ybatch in dataloader:
            ypred = pretrain_model(xbatch)
            loss = lossfn(ypred, ybatch)
            loss.backward()
            optim.step()
            lr_scheduler.step()
            print("Loss: ", loss)

        # if epoch % save_interval == 0:
        #     save_checkpoint(wandb.run.id, i, pretrain_model, optim, loss) # type: ignore
        #     # TODO: validation
        #     # pretraining_log.log({"val_loss": loss})

    # pretraining_log.finish()
