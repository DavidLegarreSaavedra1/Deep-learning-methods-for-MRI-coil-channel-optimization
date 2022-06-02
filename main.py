from NNs import *
from ROVir import *
from methods import *
from pathlib import Path as path
import cv2 as cv
import torchvision.transforms as T

WEIGHTS_PATH = (path.cwd() / "model" / "FastNN.pth").as_posix()
IMG_PATH = (path.cwd() / "data" / "test.png").as_posix()
IMG_SIZE = 144

model = load_model(WEIGHTS_PATH)

img = cv.imread(IMG_PATH, 0)
w,h = img.shape


cv.imshow("test",img)
cv.waitKey(0) 

img = T.ToTensor()(img)
img = T.Resize(IMG_SIZE)(img).unsqueeze(0)
bbox_A = model(img)[0]
print(bbox_A)
print(bbox_A.shape)
bbox_A = convert_bbox(bbox_A, w, h)
print(bbox_A)

#closing all open windows 
cv.destroyAllWindows() 