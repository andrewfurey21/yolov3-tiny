import colorsys
from PIL import ImageFont, ImageDraw, Image
from typing import List, Tuple

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
