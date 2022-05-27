from PIL import Image
from sklearn.inspection import PartialDependenceDisplay
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
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
    image = image.astype(np.float) / 255.0

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
    print(img.shape)
    cv.rectangle(
        img, (x1, y1), (x2, y2),
        (0, 255, 100), 2
    )
    #plt.figure(figsize=(10,10))
    #plt.imshow(img[:,:,::-1])
    cv.imshow("",img)

def evaluate_prediction(img, pred, gt):
    IoU = torchvision.ops.box_iou(pred, gt)
    print(IoU)
    img = cv.normalize(
        img.to('cpu').numpy(), None, 
        0, 255, cv.NORM_MINMAX,
        cv.CV_8U
    )
    img = torch.from_numpy(img).to('cuda')
    boxes = torch.zeros(2,pred.shape[1]).to('cuda')
    boxes[0] = pred
    boxes[1] = gt
    colors = ['red', 'green']
    evaluation = torchvision.utils.draw_bounding_boxes(
        img,
        boxes=boxes,
        colors=colors,
        width=2
    )

    show(evaluation)