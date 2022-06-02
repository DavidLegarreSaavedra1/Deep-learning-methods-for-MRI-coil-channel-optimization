from ..visualization import *
import torchvision
import cv2 as cv


def testing(test_path, net, device, IMG_SIZE):

    for img in test_path.iterdir():
        test = cv.imread(img.as_posix(), 0)

        processed_img = preprocess(test, device, img_size=IMG_SIZE)

        bbox_out = net(
            processed_img
        )

        print(bbox_out)

        img, bbox_out = postprocess(test, bbox_out, device)

        print(bbox_out)


        result = torchvision.utils.draw_bounding_boxes(
            img,
            bbox_out,
            colors='red',
            width=2
        )

        show(result)