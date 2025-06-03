from typing import Tuple, List
import cv2
import colorsys

def rescale_bbox_to_image():
    pass

def box_colour(class_id:int, num_classes:int) -> Tuple:
    h = float(class_id) / float(num_classes)
    r, g, b = colorsys.hsv_to_rgb(h, 1, 1)
    return (int(r * 255), int(g * 255), int(b * 255))[::-1]

def draw_bboxes(image, boxes:List[Tuple], class_ids: List[int], class_names:List[str]):
    """
        image:
        boxes: List[Tuple[x1, y1, x2, y2]]. 
        positions have to be scaled correctly to original image size (rescale_bbox_to_image)
        class_names: List[str]
    """
    assert len(boxes) == len(class_ids)

    for box, class_id in zip(boxes, class_ids):
        title = class_names[class_id]
        colour = box_colour(class_id, len(class_names))
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), colour, 4)
        cv2.putText(image, title, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)

def inference_single_image(model, input_file_name, output_file_name, class_names):
    image = cv2.imread(input_file_name)

    positions = [(50, 50, 100, 100),
               (75, 90, 180, 200),
               (300, 40, 350, 145)]
    classes = [1, 0, 1]

    draw_bboxes(image, positions, classes, class_names)
    cv2.imwrite(output_file_name, image)
