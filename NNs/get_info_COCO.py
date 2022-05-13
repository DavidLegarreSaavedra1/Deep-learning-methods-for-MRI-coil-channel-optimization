import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from PIL import Image
import os
import pydicom as dicom
import requests
import matplotlib
matplotlib.use('TkAgg')

if __name__ == '__main__':

    root = "data\heart_augmented"
    coco_annotation_file_path = os.path.join(os.getcwd(), root+"\\train.json")

    coco_annotation = COCO(annotation_file=coco_annotation_file_path)

    # Category IDs.
    cat_ids = coco_annotation.getCatIds()
    print(f"Number of Unique Categories: {len(cat_ids)}")
    print("Category IDs:")
    print(cat_ids)  # The IDs are not necessarily consecutive.

    # All categories.
    cats = coco_annotation.loadCats(cat_ids)
    cat_names = [cat["name"] for cat in cats]
    print("Categories Names:")
    print(cat_names)

    # Category ID -> Category Name.
    query_id = cat_ids[0]
    query_annotation = coco_annotation.loadCats([query_id])[0]
    query_name = query_annotation["name"]
    query_supercategory = query_annotation["supercategory"]
    print("Category ID -> Category Name:")
    print(
        f"Category ID: {query_id}, Category Name: {query_name}, Supercategory: {query_supercategory}"
    )

    # Category Name -> Category ID.
    query_name = cat_names[0]
    query_id = coco_annotation.getCatIds(catNms=[query_name])[0]
    print("Category Name -> ID:")
    print(f"Category Name: {query_name}, Category ID: {query_id}")

    # Get the ID of all the images containing the object of the category.
    img_ids = coco_annotation.getImgIds(catIds=[query_id])
    print(f"Number of Images Containing {query_name}: {len(img_ids)}")

    # Pick one image.
    img_id = img_ids[2]
    img_info = coco_annotation.loadImgs([img_id])[0]
    img_file_name = img_info["file_name"]
    # img_url = img_info["coco_url"]
    img_url = "0"
    print(
        f"Image ID: {img_id}, File Name: {img_file_name}, Image URL: {img_url}"
    )

    # Get all the annotations for the specified image.
    ann_ids = coco_annotation.getAnnIds(imgIds=[img_id], iscrowd=None)
    anns = coco_annotation.loadAnns(ann_ids)
    print(f"Annotations for Image ID {img_id}:")
    print(anns)
    print(anns[0].keys())

    ann = anns[0]
    print(ann['bbox'])

    # Use URL to load image.
   # im = Image.open(requests.get(img_url, stream=True).raw)
    im = Image.open(os.path.join(root, img_file_name))
    # Save image and its labeled version.
    plt.axis("off")
    plt.imshow(np.asarray(im), cmap='gray')

    plt.savefig(f"{img_id}.jpg", bbox_inches="tight", pad_inches=0)
    # Plot segmentation and bounding box.
    coco_annotation.showAnns(anns, draw_bbox=True)
    plt.savefig(f"{img_id}_annotated.jpg", bbox_inches="tight", pad_inches=0)

    plt.show()
