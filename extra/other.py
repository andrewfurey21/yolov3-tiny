    # TODO: transform, target_transforms, transforms, for preprocessing and data augmentation probably
    # train = torchvision.datasets.VOCDetection(
    #     root="./data/",
    #     year="2012",
    #     image_set="train",
    #     download=True,
    # )

    # bboxes = []
    # for i in range(len(train)):
    #     _, target = train[i]
    #     for obj in target['annotation']['object']:
    #         bbox = obj['bndbox']
    #         width = float(bbox['xmax']) - float(bbox['xmin'])
    #         height = float(bbox['ymax']) - float(bbox['ymin'])
    #         bboxes.append([width, height])
    #
    # bboxes = np.array(bboxes)
    # bboxes = bboxes / np.max(bboxes, axis=0)
    #
    # num_anchors = 6
    # kmeans = KMeans(n_clusters=num_anchors, random_state=0)
    # kmeans.fit(bboxes)
    #
    # anchor_boxes = kmeans.cluster_centers_
    #
    # anchor_boxes = anchor_boxes * np.max(bboxes, axis=0)
    #
    # # TODO: preprocess first, then get anchor boxes?
    # l = list(anchor_boxes)
    # l.sort(key=lambda box: box[0] * box[1])
    # anchor_boxes = np.array(l) * 416
    #
    #
    # print("Anchor Boxes:")
    # for i, anchor in enumerate(anchor_boxes):
    #     print(f"Anchor {i+1}: width={anchor[0]:.2f}, height={anchor[1]:.2f}")
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1)
