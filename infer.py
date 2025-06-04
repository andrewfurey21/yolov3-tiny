import torch
import torchvision
from PIL import Image

from yolov3tiny import data, draw, model

if __name__ == "__main__":
    torch.manual_seed(1234)

    # hyperparams
    batch_size = 1 # 32
    image_size = 416
    num_classes = 80

    names_from_paper = "./data/coco-paper.names"
    actual_names = "./data/coco.names"
    keys, indices = data.get_names(names_from_paper, actual_names)

    sample_image = "./data/val2017/000000000139.jpg"
    image, _ = data.prepare_for_inference(image_size)(Image.open(sample_image))

    yolo_model = model.YOLOv3tiny(num_classes)
    outputs = yolo_model(image.unsqueeze(0))
    print(outputs.shape)

    # pil_image = torchvision.transforms.functional.to_pil_image(image) # type: ignore
    # output = draw.draw_bboxes(pil_image, labels[..., :4], class_names)
    # output.show()
