import torch
import torchvision
from yolov3tiny import data, model, draw

# torch.set_printoptions(profile="full")

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

    lr = 0.001
    batch_size = 64
    subdivision = 2
    momentum = 0.9
    decay = 0.0005
    saturation = 1.5
    exposure = 1.5
    hue = 0.1
    epochs = 10

    # # hyperparams
    batch_size = 1
    img_size = 416
    num_classes = 1000

    # model
    pretrain_model = model.YOLOv3tinyImageNet(num_classes)
    test_input = torch.rand((2, 3, 224, 224))
    print(test_input.shape)
    test_output = pretrain_model(test_input)
    print(test_output.shape)


