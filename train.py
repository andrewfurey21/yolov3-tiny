import torch
import torchvision
from torch.utils.data import DataLoader, RandomSampler, dataloader
from yolov3tiny import data, model, draw
from yolov3tiny.loss import YOLOLoss

torch.set_printoptions(profile="full")

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

def coco_dataloader(images_dir:str, annotations_dir:str, img_size):
    names_from_paper = "./data/coco-paper.names"
    actual_names = "./data/coco.names"
    keys, _ = data.get_names(names_from_paper, actual_names)
    # dataset
    dataset = data.CocoBoundingBoxDataset(
        images=images_dir,
        annotations=annotations_dir,
        category_ids=keys,
        img_size=img_size,
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
    batch_size = 32
    img_size = 416
    num_classes = 80
    max_num_boxes= 100
    anchors = [(10, 14), (23, 27), (37, 58), (81, 82), (135, 169), (344, 319)]

    coord_weight = 5
    object_weight = 2
    no_object_weight = 0.1
    class_weight = 1
    # need a param to set number of training steps

    # dataloader
    dataloader = coco_dataloader(images_dir, annotations_dir, img_size)

    images, labels, labels_size = next(iter(dataloader))
    print("Input data shape: ", images.shape)
    print("Target labels shape: ", labels.shape)
    print("Number of labels: ", labels_size)

    # model
    yolo_v3_tiny = model.YOLOv3tiny(num_classes, anchors, img_size)

    # loss + optimizer
    lossfn = YOLOLoss(coord_weight, object_weight, no_object_weight, class_weight, max_num_boxes)

    # training loop
    output = yolo_v3_tiny(images)

    loss = lossfn(output, labels, labels_size)
    print("Loss: ", loss)

    # display_image_tensor(images[1],  labels[1], labels_size[1].item(), num_classes)


