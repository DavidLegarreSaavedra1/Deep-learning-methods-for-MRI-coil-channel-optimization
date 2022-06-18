from NNs import *
from ROVir import *
from methods import *
from pathlib import Path as path
import cv2 as cv
import torchvision.transforms as T

WEIGHTS_PATH = (path.cwd() / "model" / "net.pth").as_posix()
IMG_PATH = (path.cwd() / "data" / "input.png").as_posix()
COILS_PATH = (path.cwd() / "data")
COILS_PATH = list(COILS_PATH.glob('*.nii'))[0].as_posix()
IMG_SIZE = 96

coils = extract_coils(COILS_PATH)
model = load_model(WEIGHTS_PATH, IMG_SIZE)

w,h = coils.shape[:2]
print(w,h)

prev_img = combine_images(coils)

img_ = preprocess(prev_img)

bbox_A = model(img_)




img = img.type(torch.uint8)

cv.imshow("test2", img.numpy()[0])
training_result = torchvision.utils.draw_bounding_boxes(
    img,
    bbox_A.unsqueeze(0),
    colors='green',
    width=2
)

show(training_result)

x1,y1,x2,y2 = np.rint(bbox_A.numpy()).astype(int)


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

#closing all open windows 
plt.show()
cv.destroyAllWindows() 
