from .NNs import *
from .ROVir import *
from methods import *
from pathlib import Path as path
import cv2 as cv
import torchvision.transforms as T

WEIGHTS_PATH = (path.cwd() / "model" / "FastNN.pth").as_posix()
IMG_PATH = (path.cwd() / "data" / "test.png").as_posix()
IMG_SIZE = 96

model = load_model(WEIGHTS_PATH, IMG_SIZE)

img = cv.imread(IMG_PATH, 0)
img_np = cv.normalize(
    img, None, 0, 255,
    cv.NORM_MINMAX, cv.CV_32F
)

w,h = img.shape
print(w,h)

img = T.ToTensor()(img_np)
img_ = T.Resize(IMG_SIZE)(img).unsqueeze(0)
bbox_A = model(img_)
bbox_A = convert_bbox(bbox_A, w, h)
print(bbox_A)

img = img.type(torch.uint8)

cv.imshow("test2", img.numpy()[0])
training_result = torchvision.utils.draw_bounding_boxes(
    img,
    bbox_A.unsqueeze(0),
    colors='green',
    width=2
)

show(training_result)
plt.show()

x1,y1,x2,y2 = np.rint(bbox_A.numpy()).astype(int)
print(x1,y1,x2,y2)

mask_A = np.zeros(img_np.shape)
print(img_np.shape)
mask_A[y1:y2, x1:x2] = img_np[y1:y2, x1:x2] 

A_W = slice(x1,x2)
A_H = slice(y1,y2)
B_H = slice(80, 450)
B1_W = slice(0, int(.1*w))
B2_W = slice(int(0.9*w), -1)
regions = [A_W, A_H, B1_W, B_H, B2_W]

rovir_coils, _ = ROVir(img_np, regions, 12)

new_img = combine_images(rovir_coils)
new_img = cv.normalize(
        new_img, None, 0, 255,
        cv.NORM_MINMAX
)
plot_images(
    img_np,
    "Before ROVir",
    255,
    new_img,
    "After ROVir",
    255
)

cv.imshow("Mask", mask_A)
cv.waitKey(0)

#closing all open windows 
plt.show()
cv.destroyAllWindows() 
