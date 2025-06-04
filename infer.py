from typing import Tuple, List
import colorsys

from PIL import ImageDraw, Image, ImageFont

from yolov3tiny.model import YOLOv3tiny
from yolov3tiny import data

import torch
import torchvision

def box_colour(class_id:int, num_classes:int) -> Tuple:
    h = float(class_id) / float(num_classes)
    r, g, b = colorsys.hsv_to_rgb(h, 1, 1)
    return (int(r * 255), int(g * 255), int(b * 255))

def draw_bboxes(image:Image.Image, boxes:List[Tuple], class_names: List[str]):
    """
        image: PIL.Image.Image
        boxes: List[Tuple[x1, y1, x2, y2]].
        positions have to be scaled correctly to original image size (rescale_bbox_to_image)
        class_names: List[str]
    """

    assert len(boxes) == len(class_names), "Number of boxes don't match number of class ids"
    font = ImageFont.truetype("./fonts/Rubik-Regular.ttf")
    draw = ImageDraw.ImageDraw(image)
    for i, (box, class_name) in enumerate(zip(boxes, class_names)):
        colour = box_colour(i, len(class_names))
        x1, y1, x2, y2 = box[0], box[1], box[0] + box[2], box[1] + box[3]
        draw.rectangle((x1, y1, x2, y2), outline=colour, width=3)
        draw.text((x1, y1), class_name, font=font, fill="black")
    return image

if __name__ == "__main__":
    # hyperparams
    batch_size = 1 # 32
    image_size = 416
    num_classes = 80

    with open("./data/coco-paper.names") as f:
        paper = {line.strip(): i for i, line in enumerate(f)}

    with open("./data/coco.names") as f:
        names = [line.strip() for line in f]
        keys = {paper[name]: i for i, name in enumerate(names)}
        indices = {i: name for i, name in enumerate(names)}

    # dataset
    dataset = data.CocoBoundingBoxDataset(
        images="./data/val2017/",
        annotations="./data/annotations/instances_val2017.json",
        category_ids=keys,
        image_size=image_size,
        num_classes=num_classes
    )

    image, labels, labels_size = dataset[0]
    class_ids = torch.argmax(labels[..., 5:], dim=1).tolist()
    class_names = [indices[id] for id in class_ids]

    pil_image = torchvision.transforms.functional.to_pil_image(image) # type: ignore
    output = draw_bboxes(pil_image, labels[..., :4], class_names)
    output.show()

