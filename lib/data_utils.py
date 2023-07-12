import math

import torch.utils.data as data
import torch
from PIL import Image, ImageFilter 
import os, cv2
import numpy as np
import random
from scipy.stats import norm
from math import floor
import albumentations as A


# 垂直翻转
def random_VerticalFlip(image,target,p=0.5):
    if random.random() > p:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        if len(target) == 1:
            return image,target
        # target = np.array(target).reshape(-1, 2)
        # target = target[points_flip, :]
        target[1::2] = 1-target[1::2]
        # target[:,0] = 1-target[:,0]
        # target = target.flatten()
        return image, target
    else:
        return image, target

def random_crop(img, target,):
    """

    :param img: np格式的图片
    :param target: 归一化的坐标
    :param ratio: 随机crop的比例
    :return:
    """
    ratio = random.uniform(0.8, 0.9)
    rescale_ratio = ratio / 2
    target = np.array(target)
    if random.uniform(0,1) < 0.5:
        return img,target

    h, w = img.shape[:2]

    ctx = (0 + w) / 2
    cty = (0 + h) / 2


    # center point move
    new_ctx = ctx * random.uniform(0.90, 1.10)
    new_cty = cty * random.uniform(0.90, 1.10)

    ## random crop
    x1 = new_ctx - rescale_ratio * w
    y1 = new_cty - rescale_ratio * h

    x2 = new_ctx + rescale_ratio * w
    y2 = new_cty + rescale_ratio * h

    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(x2, w - 1)
    y2 = min(y2, h - 1)

    new_w = x2 - x1
    new_h = y2 - y1

    x1 = int(x1)
    y1 = int(y1)

    x2 = int(x2)
    y2 = int(y2)
    if len(target) == 10:
        # return to the original val
        target[::2] = target[::2] * w
        target[1::2] = target[1::2] * h
        # print(new_h, new_w)

        new_traget = target.copy()
        new_traget[::2] = (target[::2] - x1) / new_w
        new_traget[1::2] = (target[1::2] - y1) / new_h
    else:
        new_traget = target
    crop_img = img[y1:y2, x1:x2,:]


    # for i in range(5):
    #     cv2.circle(crop_img, (int(new_traget[i * 2] * new_w), int(new_traget[i * 2 + 1] * new_h)), 1, (255, 0, 0), 2)
    #     cv2.putText(crop_img, str(i), (int(new_traget[i * 2] * new_w), int(new_traget[i * 2 + 1] * new_h)), 1, 1, (0, 255, 0), 1)
    #
    # cv2.imshow("x", crop_img)
    # cv2.waitKey(0)
    return crop_img, new_traget


def random_up_down(img, MI=64,MA=128, target_size=128, p=0.5):
    if random.random() < p:
        size = np.random.randint(MI,MA)
        img = cv2.resize(img, (size,size))
        img = cv2.resize(img,(target_size,target_size))
        return img
    else:
        return cv2.resize(img,(target_size,target_size))

def random_fog(img):
    if random.random() > 0.5:
        img_f = img / 255.0
        (row, col, chs) = img.shape

        A = np.random.uniform(0.2,0.6)  # 亮度
        beta = np.random.uniform(0.0, 0.2)  # 雾的浓度
        # math.sqrt()返回数字x的平方根。
        size = math.sqrt(max(row, col))  # 雾化尺寸
        center = (row // 2, col // 2)  # 雾化中心
        for j in range(row):
            for l in range(col):
                d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
                td = math.exp(-beta * d)
                img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)
        img_f = img_f * 255.0

        return img_f.astype(np.uint8)
    else:
        return img


def random_translate(image, target):
    if random.random() > 0.7:
        image_height, image_width = image.size
        a = 1
        b = 0
        #c = 30 #left/right (i.e. 5/-5)
        c = int((random.random()-0.5) * 60)
        d = 0
        e = 1
        #f = 30 #up/down (i.e. 5/-5)
        f = int((random.random()-0.5) * 60)
        image = image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f))
        if len(target) == 10:
            target_translate = target.copy()
            target_translate = target_translate.reshape(-1, 2)
            target_translate[:, 0] -= 1.*c/image_width
            target_translate[:, 1] -= 1.*f/image_height
            target_translate = target_translate.flatten()
            target_translate[target_translate < 0] = 0
            target_translate[target_translate > 1] = 1
            return image, target_translate
        else:
            return image,target
    else:
        return image, target

def random_blur(image):
    if random.random() > 0.5:
        image = image.filter(ImageFilter.GaussianBlur(random.random()*3))
    return image

def random_occlusion(image):
    if random.random() > 0.5:
        image_np = np.array(image).astype(np.uint8)
        image_np = image_np[:,:,::-1]
        image_height, image_width, _ = image_np.shape
        occ_height = int(image_height*0.4*random.random())
        occ_width = int(image_width*0.4*random.random())
        occ_xmin = int((image_width - occ_width - 10) * random.random())
        occ_ymin = int((image_height - occ_height - 10) * random.random())
        image_np[occ_ymin:occ_ymin+occ_height, occ_xmin:occ_xmin+occ_width, 0] = int(random.random() * 255)
        image_np[occ_ymin:occ_ymin+occ_height, occ_xmin:occ_xmin+occ_width, 1] = int(random.random() * 255)
        image_np[occ_ymin:occ_ymin+occ_height, occ_xmin:occ_xmin+occ_width, 2] = int(random.random() * 255)
        image_pil = Image.fromarray(image_np[:,:,::-1].astype('uint8'), 'RGB')
        return image_pil
    else:
        return image

def  random_flip(image, target, points_flip):
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if len(target) == 1:
            return image,target
        target = np.array(target).reshape(-1, 2)
        target = target[points_flip, :]
        target[:,0] = 1-target[:,0]
        target = target.flatten()
        return image, target
    else:
        return image, target

def random_rotate(image, target, angle_max):
    if random.random() > 0.5:
        center_x = 0.5
        center_y = 0.5
        landmark_num= int(len(target) / 2)
        target_center = np.array(target) - np.array([center_x, center_y]*landmark_num)
        target_center = target_center.reshape(landmark_num, 2)
        theta_max = np.radians(angle_max)
        theta = random.uniform(-theta_max, theta_max)
        angle = np.degrees(theta)
        image = image.rotate(angle)

        if len(target) == 1:
            return image,target

        c, s = np.cos(theta), np.sin(theta)
        rot = np.array(((c,-s), (s, c)))
        target_center_rot = np.matmul(target_center, rot)
        target_rot = target_center_rot.reshape(landmark_num*2) + np.array([center_x, center_y]*landmark_num)
        return image, target_rot
    else:
        return image, target

def gen_target_pip(target, meanface_indices, target_map, target_local_x, target_local_y, target_nb_x, target_nb_y):
    num_nb = len(meanface_indices[0])
    map_channel, map_height, map_width = target_map.shape
    target = target.reshape(-1, 2)
    assert map_channel == target.shape[0]

    for i in range(map_channel):
        mu_x = int(floor(target[i][0] * map_width))
        mu_y = int(floor(target[i][1] * map_height))
        mu_x = max(0, mu_x)
        mu_y = max(0, mu_y)
        mu_x = min(mu_x, map_width-1)
        mu_y = min(mu_y, map_height-1)
        target_map[i, mu_y, mu_x] = 1
        shift_x = target[i][0] * map_width - mu_x
        shift_y = target[i][1] * map_height - mu_y
        target_local_x[i, mu_y, mu_x] = shift_x
        target_local_y[i, mu_y, mu_x] = shift_y

        for j in range(num_nb):
            nb_x = target[meanface_indices[i][j]][0] * map_width - mu_x
            nb_y = target[meanface_indices[i][j]][1] * map_height - mu_y
            target_nb_x[num_nb*i+j, mu_y, mu_x] = nb_x
            target_nb_y[num_nb*i+j, mu_y, mu_x] = nb_y

    return target_map, target_local_x, target_local_y, target_nb_x, target_nb_y


def gen_target_pip_5(target, target_map, target_local_x, target_local_y):
    map_channel, map_height, map_width = target_map.shape
    if len(target) == 10:
        target = target.reshape(-1, 2)
        assert map_channel == target.shape[0]

        for i in range(map_channel):
            mu_x = int(floor(target[i][0] * map_width))
            mu_y = int(floor(target[i][1] * map_height))
            mu_x = max(0, mu_x)
            mu_y = max(0, mu_y)
            mu_x = min(mu_x, map_width-1)
            mu_y = min(mu_y, map_height-1)
            target_map[i, mu_y, mu_x] = 1
            shift_x = target[i][0] * map_width - mu_x
            shift_y = target[i][1] * map_height - mu_y
            target_local_x[i, mu_y, mu_x] = shift_x
            target_local_y[i, mu_y, mu_x] = shift_y

    return target_map, target_local_x, target_local_y

class ImageFolder_pip(data.Dataset):
    def __init__(self, root, imgs, input_size, num_lms, net_stride, points_flip, meanface_indices, transform=None, target_transform=None):
        self.root = root
        self.imgs = imgs
        self.num_lms = num_lms
        self.net_stride = net_stride
        self.points_flip = points_flip
        self.meanface_indices = meanface_indices
        self.num_nb = len(meanface_indices[0])
        self.transform = transform
        self.target_transform = target_transform
        self.input_size = input_size

    def __getitem__(self, index):

        img_name, target = self.imgs[index]

        img = Image.open(os.path.join(self.root, img_name)).convert('RGB')
        img, target = random_translate(img, target)
        # img = random_occlusion(img)
        img, target = random_flip(img, target, self.points_flip)
        img, target = random_rotate(img, target, 30)
        img = random_blur(img) # close blur

        target_map = np.zeros((self.num_lms, int(self.input_size/self.net_stride), int(self.input_size/self.net_stride)))
        target_local_x = np.zeros((self.num_lms, int(self.input_size/self.net_stride), int(self.input_size/self.net_stride)))
        target_local_y = np.zeros((self.num_lms, int(self.input_size/self.net_stride), int(self.input_size/self.net_stride)))
        # target_nb_x = np.zeros((self.num_nb*self.num_lms, int(self.input_size/self.net_stride), int(self.input_size/self.net_stride)))
        # target_nb_y = np.zeros((self.num_nb*self.num_lms, int(self.input_size/self.net_stride), int(self.input_size/self.net_stride)))
        target_map, target_local_x, target_local_y, target_nb_x, target_nb_y = gen_target_pip_5(target, target_map, target_local_x, target_local_y)
        
        target_map = torch.from_numpy(target_map).float()
        target_local_x = torch.from_numpy(target_local_x).float()
        target_local_y = torch.from_numpy(target_local_y).float()
        target_nb_x = torch.from_numpy(target_nb_x).float()
        target_nb_y = torch.from_numpy(target_nb_y).float()

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target_map = self.target_transform(target_map)
            target_local_x = self.target_transform(target_local_x)
            target_local_y = self.target_transform(target_local_y)
            target_nb_x = self.target_transform(target_nb_x)
            target_nb_y = self.target_transform(target_nb_y)

        return img, target_map, target_local_x, target_local_y, target_nb_x, target_nb_y

    def __len__(self):
        return len(self.imgs)

def letter_bbox(img, target, target_size):
    np_img = np.array(img)
    h,w = np_img.shape[:2]
    im_ratio = h / w
    if im_ratio > 1:
        new_h = target_size
        new_w = int(new_h / im_ratio)
        ratio = new_w / target_size
        if len(target) == 10:
            target[::2] = target[::2] * ratio
    else:
        new_w = target_size
        new_h = int(new_w * im_ratio)
        ratio = new_h / target_size
        if len(target) == 10:
            target[1::2] = target[1::2] * ratio
    np_img = cv2.resize(np_img, (new_w, new_h))
    det_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    det_img[:new_h, :new_w, :] = np_img

    img = Image.fromarray(det_img)
    return img, target


def letterbox(ori_img, target, new_shape=(128, 128), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    img = np.array(ori_img)
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 128), np.mod(dh, 128)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border


    ratio = (new_unpad[0] / new_shape[0], new_unpad[1] / new_shape[1] )

    if dw:
        # x = target[::2]
        target[::2] = target[::2] * ratio[0] + dw / new_shape[0]

    else:
        # y = target[1::2]
        # y = y * ratio[0] + dh / 128
        # x = target[::2]
        target[1::2] = target[1::2] * ratio[1] + dh / new_shape[1]

    return Image.fromarray(img), target

class ImageFolder_pip_5(data.Dataset):
    def __init__(self, root, imgs, input_size, num_lms, net_stride, points_flip, transform=None,
                 target_transform=None):
        self.root = root
        self.imgs = imgs
        self.num_lms = num_lms
        self.net_stride = net_stride
        self.points_flip = points_flip
        self.transform = transform
        self.target_transform = target_transform
        self.input_size = input_size

    def __getitem__(self, index):

        img_name, target = self.imgs[index]

        img = Image.open(os.path.join(self.root, img_name)).convert('RGB')

        # img = random_occlusion(img)
        # if len(target) != 10:
        # img, target = letter_bbox(img, target, self.input_size)
        # img, target = letterbox(img, target, auto=False)
        
        ## TODO  128->64 ................................................................
        img, target = random_crop(np.array(img), target)
        # img = cv2.resize(np.array(img),(64,64))
        img = random_up_down(img, MI=32, MA=64, target_size=64)
        # img = random_fog(img)
        img = Image.fromarray(img)
        # img, target = random_translate(img, target)
        img, target = random_flip(img, target, self.points_flip)
        # img, target = random_VerticalFlip(img, target)
        img, target = random_rotate(img, target, 10)
        # img = random_blur(img)

        # img_show = np.array(img)
        #
        # h, w = img_show.shape[:2]
        # if len(target) == 10:
        #     for i in range(5):
        #         cv2.circle(img_show, (int(target[i * 2] * w), int(target[i * 2 + 1] * h)), 1, (0, 0, 255), 2)
        #         cv2.putText(img_show, str(i), (int(target[i * 2] * w), int(target[i * 2 + 1] * h)), 1, 1, (0, 0, 255), 2)
        # # cv2.imshow("x", img_show)
        # # cv2.waitKey(0)
        # cv2.imwrite("{}.jpg".format(random.randint(0,1000)), img_show[:,:,::-1])
        # return
        # cv2.imshow("x", im_)
        # cv2.waitKey(0)

        # target_map = np.zeros(
        #     (self.num_lms, int(self.input_size / self.net_stride), int(self.input_size / self.net_stride)))
        # target_local_x = np.zeros(
        #     (self.num_lms, int(self.input_size / self.net_stride), int(self.input_size / self.net_stride)))
        # target_local_y = np.zeros(
        #     (self.num_lms, int(self.input_size / self.net_stride), int(self.input_size / self.net_stride)))
        #
        # # target_map, target_local_x, target_local_y, target_nb_x, target_nb_y = gen_target_pip_5(target,
        # #                                                                                       target_map,
        # #                                                                                       target_local_x,
        # #                                                                                       target_local_y)
        # # widerface
        # target_map, target_local_x, target_local_y = gen_target_pip_5(target,
        #                                                                     target_map,
        #                                                                     target_local_x,
        #                                                                     target_local_y)
        # target_map = torch.from_numpy(target_map).float()
        # target_local_x = torch.from_numpy(target_local_x).float()
        # target_local_y = torch.from_numpy(target_local_y).float()
        #
        #
        #
        # if self.transform is not None:
        #     img = self.transform(img)
        # if self.target_transform is not None:
        #     target_map = self.target_transform(target_map)
        #     target_local_x = self.target_transform(target_local_x)
        #     target_local_y = self.target_transform(target_local_y)
        #
        # label = np.array([0])
        # if len(target) == 10:
        #     label += 1
        #
        #
        # return img, target_map, target_local_x, target_local_y, label
        # im_ = np.array(img)
        # for i in range(5):
        #     cv2.circle(im_, (int(target[i*2] * 128), int(target[i*2+1] * 128)), 1, (0, 0, 255), 2)
        #     cv2.putText(im_, str(i), (int(target[i * 2] * 128), int(target[i * 2 + 1] * 128)), 1, 1, (0, 0, 255), 2)
        #
        # cv2.imshow("x", im_)
        # cv2.waitKey(0)

        if self.transform is not None:
            img = self.transform(img)

        label = np.array([1])
        if len(target) != 10:
            target = np.zeros(10)
            label -= 1
        return img_name, img, target, label

    def __len__(self):
        return len(self.imgs)


class ImageFolder_pip_5_blurness(data.Dataset):
    """
      增加了blurness预测的一维
    """
    def __init__(self, root, imgs, input_size, num_lms, net_stride, points_flip, transform=None,
                 target_transform=None):
        self.root = root
        self.imgs = imgs
        self.num_lms = num_lms
        self.net_stride = net_stride
        self.points_flip = points_flip
        self.transform = transform
        self.target_transform = target_transform
        self.input_size = input_size
        self.T = A.Compose([
                A.GaussianBlur(p=1.0,blur_limit=(3,5)),
                A.MultiplicativeNoise(p=1.0),
                A.RandomBrightnessContrast(p=1.0),
                A.ISONoise(p=1.0),
                # A.HueSaturationValue(val_shift_limit=20,p=1.0),
                A.GaussNoise(p=1.0,var_limit=(30.0, 50.0)),
                A.MotionBlur(p=1.0, blur_limit=(10, 17))]
                )

    def __getitem__(self, index):

        img_name, target = self.imgs[index]

        img = Image.open(os.path.join(self.root, img_name)).convert('RGB')

        if len(target) != 1 or target != 2:
            r = random.random()
            if r < 0.3:
                target = np.array([2.0])
                img, target = random_crop(np.array(img), target)
                img = random_up_down(img, MI=32, MA=48, target_size=64)
                img = self.T(image=img)['image']
                # cv2.imshow('x', img)
                # cv2.waitKey(0)
                # cv2.imwrite('x1.jpg',img)
                img = Image.fromarray(img)

            else:
                ## TODO  128->64 ................................................................
                img, target = random_crop(np.array(img), target)
                # img = cv2.resize(np.array(img),(64,64))
                img = random_up_down(img, MI=48, MA=64, target_size=64)
                # img = random_fog(img)
                img = Image.fromarray(img)
                # img, target = random_translate(img, target)
        else:
            img = np.array(img)
            img = cv2.resize(img, (64,64))
            img = Image.fromarray(img)
        img, target = random_flip(img, target, self.points_flip)
        # img, target = random_VerticalFlip(img, target)
        img, target = random_rotate(img, target, 10)

        if self.transform is not None:
            img = self.transform(img)

        if len(target) != 10:
            label = np.array([0]) if target == -1 else np.array([2])
            target = np.zeros(10)

        else:
            label = np.array([1])
        return img_name, img, target, label

    def __len__(self):
        return len(self.imgs)



if __name__ == '__main__':
    pass
    
