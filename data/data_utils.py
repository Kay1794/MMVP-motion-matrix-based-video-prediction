import numpy as np


def CenterCrop(img,crop_w,crop_h):
    center = np.array(img.shape[:2]) / 2
    x = center[1] - crop_w / 2
    y = center[0] - crop_h / 2

    crop_img = img[int(y):int(y + crop_h), int(x):int(x + crop_w)]
    return crop_img

