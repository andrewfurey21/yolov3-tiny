import torch
import torchvision
from yolov3tiny import data, draw

if __name__ == "__main__":
    torch.manual_seed(1234)

    # hyperparams
    batch_size = 1 # 32
    image_size = 416
    num_classes = 80

    names_from_paper = "./data/coco-paper.names"
    actual_names = "./data/coco.names"
    keys, indices = data.get_names(names_from_paper, actual_names)

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

