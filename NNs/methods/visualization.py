from PIL import Image
import torch
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import matplotlib
import torchvision
matplotlib.use('TkAgg')

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = T.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def preprocess(img, img_size = 512):

    image = cv.resize(img, (img_size, img_size))
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = image.astype("float") / 255.0

    image = np.expand_dims(image, axis=0)
    return image

def postprocess(results):
    h, w = 512, 512

    x1, y1, width, height = results[0]

    x1 = int(w*x1)
    y1 = int(h*y1)
    x2 = int(w*width)
    y2 = int(h*height)

    return (x1,y1,x2,y2)

def draw_bbox(img, bbox):
    x1, y1, x2, y2 = bbox
    cv.rectangle(
        img, (x1, y1), (x2, y2),
        (0, 255, 100), 2
    )
    plt.figure(figsize=(10,10))
    plt.imshow(img[:,:,::-1])