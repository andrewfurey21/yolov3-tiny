from yolov3tiny import data

if __name__ == "__main__":
    # hyperparams
    batch_size = 1 # 32
    image_size = 416
    num_classes = 80

    with open("./data/coco-paper.names") as f:
        paper = {line.strip(): i for i, line in enumerate(f)}

    with open("./data/coco.names") as f:
        keys = {paper[line.strip()]: (i, line.strip()) for i, line in enumerate(f)}

    print(keys)

    # dataset
    dataset = data.CocoBoundingBoxDataset(
        images="./data/val2017/",
        annotations="./data/annotations/instances_val2017.json",
        category_ids=keys,
        image_size=image_size,
        num_classes=num_classes
    )

    print(dataset[0])

    # collate_fn = data.CollateCOCO("./data/coco.names", image_size)
    # dataloader = DataLoader(dataset,
    #                         batch_size=batch_size,
    #                         shuffle=True,
    #                         num_workers=1,
    #                         collate_fn=collate_fn)
    #
    # image = next(iter(dataloader))
    #
    # pil_image = torchvision.transforms.ToPILImage()(image.squeeze(0))
    # pil_image.show()






