from PIL import Image
import torch
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import matplotlib
import torchvision
matplotlib.use('TkAgg')



def plot_img_mask(img, bbox):
    print(bbox.shape)
    img = img[0].cpu().numpy().transpose(1, 2, 0)
    x0, y0, x1, y1 = bbox[0].numpy()

    print(np.max(img))
    
    print(type(img))
    print(img.shape)
    print(x0)
    cv.rectangle(
        img, (int(x0), int(y0)),
        (int(x1), int(y1)),
        (0, 255, 0), 3
    )
    

    cv.imwrite(img, "test")
    #im = plt.imshow(img, cmap='gray')
    #plt.show()

def plot_img_bbox(img, bbox):
    img = Image.fromarray(img)
    torchvision.utils.draw_bounding_boxes(img, bbox)
