import torch
import torchvision
from torch.utils.data import DataLoader, RandomSampler, dataloader
from yolov3tiny import data, model, draw

def display_image_tensor(image: torch.Tensor, labels:torch.Tensor, size:int, num_classes:int):
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

def build_dataloader(images_dir:str, annotations_dir:str):
    names_from_paper = "./data/coco-paper.names"
    actual_names = "./data/coco.names"
    keys, _ = data.get_names(names_from_paper, actual_names)
    # dataset
    dataset = data.CocoBoundingBoxDataset(
        images=images_dir,
        annotations=annotations_dir,
        category_ids=keys,
        image_size=image_size,
        num_classes=num_classes,
        max_num_boxes=max_num_boxes
    )

    # dataloading
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            sampler=sampler,
                            collate_fn=data.collate_coco_sample)
    return dataloader


if __name__ == "__main__":
    torch.manual_seed(12345)

    # dataset

    images_dir = "./data/val2017/"
    annotations_dir = "./data/annotations/instances_val2017.json"

    # hyperparams
    batch_size = 1 # 32
    image_size = 416
    num_classes = 80
    max_num_boxes= 100
    anchors = [(10, 14), (23, 27), (37, 58), (81, 82), (135, 169), (344, 319)]
    ignore_thresh = 0.5
    no_object_coeff = .5
    coord_cooef = 5
    # need way to set number of training steps

    # dataloader
    dataloader = build_dataloader(images_dir, annotations_dir)

    images, labels, labels_size = next(iter(dataloader))
    print("Input data shape: ", images.shape)
    print("Target labels shape: ", labels.shape)
    print("Number of labels: ", labels_size)

    yolo_v3_tiny = model.YOLOv3tiny(num_classes)
    output = yolo_v3_tiny(images)

    display_image_tensor(images[0], labels[0], labels_size.item(), num_classes)


