import torch
from torch.utils.data import DataLoader, RandomSampler
from yolov3tiny import data, model, loss

if __name__ == "__main__":
    torch.manual_seed(1234)

    # hyperparams
    batch_size = 32 # 32
    image_size = 416
    num_classes = 80
    max_num_boxes= 100

    names_from_paper = "./data/coco-paper.names"
    actual_names = "./data/coco.names"
    keys, indices = data.get_names(names_from_paper, actual_names)

    # dataset
    dataset = data.CocoBoundingBoxDataset(
        images="./data/val2017/",
        annotations="./data/annotations/instances_val2017.json",
        category_ids=keys,
        image_size=image_size,
        num_classes=num_classes,
        max_num_boxes=max_num_boxes
    )

    # image, labels, labels_size = dataset[0]
    # class_ids = torch.argmax(labels[..., 5:], dim=1).tolist()
    # class_names = [indices[id] for id in class_ids]

    # dataloading
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            sampler=sampler,
                            collate_fn=collate_fn)

    image, labels, labels_size = next(iter(dataloader))
    print(image.shape, labels.shape, labels_size.shape)


