import collections

import onnxruntime
import torch
from tqdm import tqdm

from lib.networks import Pip_mbnetv3_small,Pip_mbnetv2_small
from lib.mobilenetv3 import mobilenetv3_small, mobilenetv3_small_light
from lib.mobilenetv2 import mobilenet_v2
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
transformer = transforms.Compose([transforms.ToTensor(),
                  normalize])
input_name = 'input'
output_names = ['x_pred', 'y_pred', 'score']

def letter_bbox(img, target_size=128):
    np_img = np.array(img)
    h,w = np_img.shape[:2]
    im_ratio = h / w
    if im_ratio > 1:
        new_h = target_size
        new_w = int(new_h / im_ratio)

    else:
        new_w = target_size
        new_h = int(new_w * im_ratio)

    np_img = cv2.resize(np_img, (new_w, new_h))
    det_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    det_img[:new_h, :new_w, :] = np_img

    img = Image.fromarray(det_img)
    return img

def getMaxPos(x):
    return torch.argmax(torch.max(x,1).values,0).item(),torch.argmax(torch.max(x,0).values,0).item()

def pred(model, img_tensor):
    with torch.no_grad():
        x,y,score = model(img_tensor)
        x = x.squeeze()
        y = y.squeeze()
        score = 1 / (1+torch.exp(-score))
        landmarks = []

    return score


def get_imgs(txt_path):
    imgs = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        imgs.append(line.split()[0])
    return imgs

def pred_imgs(imgs, root, model, save_path):
    f = open(save_path, 'w')
    for img_p in tqdm(imgs):
        img_path = root + "/" + img_p
        # print(img_p)
        img = Image.open(img_path).convert('RGB')
        # img = letter_bbox(img)
        img = cv2.resize(np.array(img),(128,128))
        img = Image.fromarray(img)
        tensor = transformer(img).unsqueeze(0).cuda()
        score = pred(model,tensor).cpu().squeeze().numpy()
        f.write(img_p + " " + str(score) + "\n")

def pred_imgs_onnx_withTxt(imgs, root, model, save_path, size = (128,128)):
    input_size = size
    f = open(save_path, 'w')
    for img_p in tqdm(imgs):
        img_path = root + "/" + img_p
        img = cv2.imread(img_path)
        img_processed = cv2.resize(img, (128, 128))
        # img_processed, ratio = letter_bbox(img)
        blob = cv2.dnn.blobFromImage(img_processed, 1.0 / 127.5, input_size, (127.5, 127.5, 127.5), swapRB=True)
        net_outs = model.run(output_names, {input_name: blob})
        x_pred, y_pred, score = net_outs
        score = np.squeeze(score)
        if len(score) == 1:
            score = 1 / (1 + np.exp(-score))
        elif len(score) == 2:
            score = 1 / (1 + np.exp(-score))[0]

        if len(x_pred.shape) == 3:
            x_pred = x_pred[0, :, 0]
            y_pred = y_pred[0, :, 0]
        elif len(x_pred.shape) == 2:
            x_pred = x_pred[0,:]
            y_pred = y_pred[0,:]
        h, w = img.shape[:2]
        # for i in range(5):
            # cv2.putText(img, str(i),
            #             (int(x_pred[i] * w), int(y_pred[i] * h)), 1, 1,
            #             (0, 0, 255), 2)
            # cv2.circle(img, (int(x_pred[i] * w), int(y_pred[i] * h)),pip
            #            1, (0, 0, 255), 2)
        # cv2.imshow('x', img)
        # cv2.waitKey(0)

        f.write(img_p + " " + str(float(score)) + " ")
        for i in range(5):
            f.write(str(x_pred[i]) + " " + str(y_pred[i]) + " ")
        f.write("\n")
    f.close()


def pred_imgs_onnx(imgs, root, model, save_path, size = (128,128)):
    input_size = size
    pred_mps = collections.defaultdict(list)
    for img_p in tqdm(imgs):
        img_path = root + "/" + img_p
        img = cv2.imread(img_path)
        img_processed = cv2.resize(img, (128, 128))
        # img_processed, ratio = letter_bbox(img)
        blob = cv2.dnn.blobFromImage(img_processed, 1.0 / 127.5, input_size, (127.5, 127.5, 127.5), swapRB=True)
        net_outs = model.run(output_names, {input_name: blob})
        x_pred, y_pred, score = net_outs
        score = np.squeeze(score)
        if len(score) == 1:
            score = 1 / (1 + np.exp(-score))
        elif len(score) == 2:
            score = 1 / (1 + np.exp(-score))[0]

        if len(x_pred.shape) == 3:
            x_pred = x_pred[0, :, 0]
            y_pred = y_pred[0, :, 0]
        elif len(x_pred.shape) == 2:
            x_pred = x_pred[0,:]
            y_pred = y_pred[0,:]
        h, w = img.shape[:2]
        # for i in range(5):
            # cv2.putText(img, str(i),
            #             (int(x_pred[i] * w), int(y_pred[i] * h)), 1, 1,
            #             (0, 0, 255), 2)
            # cv2.circle(img, (int(x_pred[i] * w), int(y_pred[i] * h)),pip
            #            1, (0, 0, 255), 2)
        # cv2.imshow('x', img)
        # cv2.waitKey(0)
        pred_mps[img_p] = [score]
        for i in range(5):
            pred_mps[img_p].append(x_pred[i])
            pred_mps[img_p].append(y_pred[i])
    return pred_mps


if __name__ == '__main__':
    # model_path = "../snapshots/Widerface/pipnet_mbv3_v5_64_COLORAUG/epoch120.pth"
    # flag = 3  # 2 3
    # if flag == 3:
    #     ## v3
    #     model_path = '../snapshots/Widerface/pipnet_mbv3_v5_6_14_COLORAUG/epoch50.pth'
    #     mbnet = mobilenetv3_small()
    #     model = Pip_mbnetv3_small(mbnet=mbnet).cuda()
    #     model_dict = torch.load(model_path, map_location='cpu')
    #     model.load_state_dict(model_dict)
    #     model.eval()
    #
    #
    # if flag == 2:
    #     ## v2
    #     mbnet = mobilenet_v2(width_mult=0.75)
    #     model = Pip_mbnetv2_small(mbnet=mbnet).cuda()
    #     model_dict = torch.load(model_path, map_location='cpu')
    #     model.load_state_dict(model_dict)
    #     model.eval()


    # onnx
    onnx_path = "../onnx/pipnet_mbv1_blurness_v5_64_retinaface_100ep.onnx"
    onnx_model = onnxruntime.InferenceSession(onnx_path, None)

    img_path = '/opt/data/face_landmark/eval_data/eval_ipc_labels.txt'
    # img_path = '/opt/data/face_landmark/ipc_6_14_labels.txt'
    img_root = '/opt/data/face_landmark/eval_data'  # '/opt/data/face_landmark/capture_pic_6_14'
    # img_root = '/opt/data/face_landmark/capture_pic_6_14'
    

    imgs = get_imgs(img_path)

    # pred_imgs(imgs, img_root, model, "../TXT/eval_face_imgs_64.txt")
    pred_imgs_onnx_withTxt(imgs, img_root, onnx_model, "../TXT/eval_retinaface_blurness_100ep.txt", size=(64,64))