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

def transform_bbox(bbox):
    x1,y1, width, height = bbox

    return x1, y1, x1+width, y1+height

def preprocess(img, device = "cpu", img_size = 144):
    """Preprocess images to prepare for model    

    img_size: Size of the image to resize
    """

    image = cv.resize(img, (img_size, img_size))
    #image = image.astype(np.float) / 255.0

    # convert image to a tensor
    image = torch.from_numpy(
        image
    ).to(device).float()
    image = image.float()
    hist, bins = torch.histogram(image.cpu(),bins=64)
    limit = torch.quantile(bins, 0.99)
    image = image/limit

    # Unsqueeze tensor to add batch dimension
    image = image.unsqueeze(0)
    image = image.unsqueeze(0)

    # Normalize the batch
    image = T.Normalize(
        mean=image.mean(),
        std=image.std()
    )(image)

    return image

def postprocess(img, bbox, device):
    """Postprocess the output

    This function will adapt the output of the network
    to the original dimensions of the image
    """

    img = torch.from_numpy(img).unsqueeze(0)
    #img = T.Resize(size=w)(img)

    img = img.type(torch.uint8)


    print(img.shape)
    w, h = img.shape[-2:]
    print(w,h)
    x1, y1, width, height = bbox[0]

    x1 = int(w*x1)
    y1 = int(h*y1)
    x2 = int(w*width)
    y2 = int(h*height)

    new_bbox = [x1,y1,x2,y2]
    new_bbox = torch.tensor(
       new_bbox, device=device 
    ).unsqueeze(0)

    return img, new_bbox

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