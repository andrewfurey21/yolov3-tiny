import torch
from yolov3tiny import data
from tqdm import trange
import numpy as np
import mlpack

from matplotlib import pyplot as plt

def get_coco_anchor_boxes(images_dir:str, annotations_dir:str, img_size:int, num_classes:int, max_num_boxes:int):
    wh = []
    dataloader = data.build_coco_dataloader(images_dir, annotations_dir, img_size, num_classes, max_num_boxes, 1, False, data.prepare_for_inference)

    for _ in trange(len(dataloader)):
        batch = next(iter(dataloader))
        boxw = (batch[1][0, :, 2] - batch[1][0, :, 0]).long().tolist()[:batch[2][0]]
        boxh = (batch[1][0, :, 3] - batch[1][0, :, 1]).long().tolist()[:batch[2][0]]
        wh.extend(list(zip(boxw, boxh)))
    wh_np = np.array(wh)

    cluster = mlpack.kmeans(input_=wh_np, clusters=6, algorithm="naive")
    centroids = cluster["centroid"].astype(int)
    outputs = cluster["output"].astype(int)
    return sorted(centroids, key=lambda wh: wh[0] * wh[1]), sorted(outputs, key=lambda x: x[2])

def plot(outputs:np.ndarray):
    x = outputs[:, 0]
    y = outputs[:, 1]
    classes = outputs[:, 2].astype(int)
    plt.scatter(x, y, c=classes, cmap='tab10', s=100, edgecolors='k')
    plt.xlabel("W")
    plt.ylabel("H")
    plt.title("clustered anchor boxes")
    plt.colorbar(label="Centroid")
    plt.show()

if __name__ == "__main__":
    torch.manual_seed(12345)
    val_annotations_dir = "./data/annotations/instances_val2017.json"
    val_images_dir = "./data/val2017"
    img_size = 416
    num_classes = 80
    batch_size = 1
    max_num_boxes = 100

    output_file = "anchors"

    centroids, outputs = get_coco_anchor_boxes(val_images_dir, val_annotations_dir, img_size, num_classes, max_num_boxes)
    with open(output_file, 'w') as f:
        f.write("[ ")
        for i, (w, h) in enumerate(centroids):
            f.write(f"({w}, {h})")
            if i != len(centroids) - 1:
                f.write(" , ")
        f.write(" ]\n")

    # plot(np.array(outputs))



