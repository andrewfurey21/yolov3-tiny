import torch
import torchvision
from yolov3tiny import data, draw

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
    output = draw.draw_bboxes(pil_image, labels[..., :4], class_names)
    output.show()

