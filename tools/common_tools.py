import numpy as np
import cv2
from PIL import Image

def letter_bbox(img, target_size=128):
    h,w = img.shape[:2]
    im_ratio = h / w
    if im_ratio > 1:
        # 占H比列不变，W比例变小
        new_h = target_size
        new_w = int(new_h / im_ratio)
        ratio = [new_w / target_size, 1]
    else:
        # 与上相反
        new_w = target_size
        new_h = int(new_w * im_ratio)
        ratio = [1, new_h / target_size]
    np_img = cv2.resize(img, (new_w, new_h))
    det_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    det_img[:new_h, :new_w, :] = np_img

    return det_img, ratio

def letter_bbox_PIL(img, target_size=128):
    np_img = np.array(img)
    h,w = np_img.shape[:2]
    im_ratio = h / w
    if im_ratio > 1:
        new_h = target_size
        new_w = int(new_h / im_ratio)
        ratio = [new_w / target_size, 1]

    else:
        new_w = target_size
        new_h = int(new_w * im_ratio)
        ratio = [1, new_h / target_size]

    np_img = cv2.resize(np_img, (new_w, new_h))
    det_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    det_img[:new_h, :new_w, :] = np_img

    img = Image.fromarray(det_img)
    return img, ratio