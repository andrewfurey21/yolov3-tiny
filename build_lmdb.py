from yolov3tiny import data
import torchvision
import pathlib

if __name__ == "__main__":
    dataset_path = pathlib.Path("../imagenet_coco_training_data/ILSVRC/Data/CLS-LOC/")
    split = "train"

    img_size = 224
    brightness = 0.5
    contrast = 0.5
    saturation = 0.5
    hue = 0.5
    transform = data.prepare_for_imagenet_training(img_size, brightness, contrast, saturation, hue)
    print("Building dataset......", end="", flush=True)
    dataset = torchvision.datasets.ImageFolder(root=(dataset_path/split), transform=transform)
    print("built.")

    data.convert_to_lmdb(dataset=dataset, split=split, save_to="../imagenet_coco_training_data/lmdb/train.lmdb")

