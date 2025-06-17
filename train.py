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



if __name__ == "__main__":
    torch.manual_seed(12345)

    # dataset
    val_images_dir = "./data/val2017/"
    val_annotations_dir = "./data/annotations/instances_val2017.json"

    # hyperparams
    batch_size = 1
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
    dataloader = data.build_coco_dataloader(val_images_dir, val_annotations_dir, img_size, num_classes, max_num_boxes, batch_size, True, data.prepare_for_training)

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

    display_image_tensor(images[0],  labels[0], labels_size[0].item(), num_classes)


