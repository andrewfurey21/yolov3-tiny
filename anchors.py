import torch
from yolov3tiny import data
from tqdm import trange
import numpy as np
import mlpack

def get_coco_anchor_boxes(images_dir:str, annotations_dir:str, img_size:int, num_classes:int, max_num_boxes:int):
    wh = []
    dataloader = data.build_coco_dataloader(images_dir, annotations_dir, img_size, num_classes, max_num_boxes, 1, False, data.prepare_for_inference)

    for _ in trange(len(dataloader)):
        batch = next(iter(dataloader))
        boxw = (batch[1][0, :, 2] - batch[1][0, :, 0]).long().tolist()[:batch[2][0]]
        boxh = (batch[1][0, :, 3] - batch[1][0, :, 1]).long().tolist()[:batch[2][0]]
        wh.extend(list(zip(boxw, boxh)))
    wh_np = np.array(wh)
    return mlpack.kmeans(input_=wh_np, clusters=6, algorithm="naive")["centroid"].astype(int)

if __name__ == "__main__":
    torch.manual_seed(12345)
    val_annotations_dir = "./data/annotations/instances_val2017.json"
    val_images_dir = "./data/val2017"
    img_size = 416
    num_classes = 80
    batch_size = 1
    max_num_boxes = 100

    output_file = "anchors"

    anchors = get_coco_anchor_boxes(val_images_dir, val_annotations_dir, img_size, num_classes, max_num_boxes)
    with open(output_file, 'w') as f:
        f.write("[ ")
        for i, (w, h) in enumerate(anchors):
            f.write(f"({w}, {h})")
            if i != len(anchors) - 1:
                f.write(" , ")
        f.write(" ]\n")



