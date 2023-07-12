import onnxruntime
import os
import numpy as np
import cv2
from PIL import Image
from common_tools import letter_bbox, letter_bbox_PIL
from tqdm import tqdm
import torch
from lib.mobilenetv3 import mobilenetv3_small, mobilenetv3_small_light
from lib.mobilenetv2 import mobilenet_v2

from lib.networks import Pip_mbnetv3_small,Pip_mbnetv2_small,Pip_mbnetv3_small_ld
import torchvision.transforms as transforms
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])

transformer = transforms.Compose([transforms.ToTensor(),
                  normalize])

def pipnet_onnx_pred(model, imgs, root, targets, size=(128,128)):
    x_loss = 0
    y_loss = 0
    input_size = size
    for im in tqdm(imgs):
        target = targets[im]
        x_t = target[::2]
        y_t = target[1::2]
        img = cv2.imread(root + im)
        img_processed = cv2.resize(img, size)
        # img_processed, ratio = letter_bbox(img)
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img_processed, 1.0 / 127.5, input_size, (127.5, 127.5, 127.5), swapRB=True)
        net_outs = model.run(output_names, {input_name: blob})
        x_pred, y_pred, score = net_outs
        score = np.squeeze(score)
        if len(score) == 1:
            cls = 1 / (1 + np.exp(-score))

        else:
            cls = 1 / (1 + np.exp(-score[0]))
            blur = 1 / (1 + np.exp(-score[1]))
        # if score < 0.3:
        #     continue
        if len(x_pred.shape) == 3:
            x_pred = x_pred[0, :, 0]
            y_pred = y_pred[0, :, 0]
        elif len(x_pred.shape) == 2:
            x_pred = x_pred[0, :]
            y_pred = y_pred[0, :]

        # 还原
        x_s = x_pred * 1
        y_s = y_pred * 1

        x_loss += np.mean(np.abs(x_s - x_t))
        y_loss += np.mean(np.abs(y_s - y_t))
            # 还原图像
            # x_s = x_pred[i] * h * ratio[0]
            # y_s = y_pred[i] * w * ratio[1]
    total = len(image_ls)
    x_loss = x_loss / total
    y_loss = y_loss / total
    print("**************************   ONNX  *********************")
    print("onnx x_loss:", x_loss)
    print("onnx y_loss:", y_loss)
        #     cv2.circle(img, (int(x_s), int(y_s)), 1, (0, 0, 255), 2)
        #     cv2.putText(img, str(i), (int(x_s), int(y_s)), 1, 1, (0, 0, 255), 2)
        # cv2.imshow("x", img)
        # cv2.waitKey(0)

def letterbox(ori_img, target=None, new_shape=(128, 128), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
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
    try:
        if target.any():
            return Image.fromarray(img), target
    except:
        return Image.fromarray(img), ratio

def pipnet_pt_pred(model, imgs, root, targets, size=(128,128)):
    x_loss = 0
    y_loss = 0

    with torch.no_grad():
        for im in tqdm(imgs):
            target = targets[im]
            img = Image.open(root + im)
            img_show = np.array(img)
            img = img.convert('RGB')
            # img, ratio = letter_bbox_PIL(img)
            # img, target = letterbox(img,target)
            #TODO resize
            img = cv2.resize(np.array(img),size)
            img = Image.fromarray(img)
            x_t = target[::2]
            y_t = target[1::2]
            tensor = transformer(img).cuda()
            x_pred, y_pred, score = model(tensor .unsqueeze(0))
            if len(score) == 1:
                cls = 1 / (1+ torch.exp(-score))

            else:
                cls = 1 / (1+ torch.exp(-score[0]))
                blur = 1 / (1 + torch.exp(-score[1]))
            # if score < 0.3 :
            #     continue
            if len(x_pred.shape) == 3:
                x_pred = x_pred[0,:,0].cpu().numpy()
                y_pred = y_pred[0,:,0].cpu().numpy()
            elif len(x_pred.shape) == 2:
                x_pred = x_pred[0, :].cpu().numpy()
                y_pred = y_pred[0, :].cpu().numpy()

            #
            # img = np.array(img)
            # for i in range(5):
            #     cv2.putText(img, str(i), (int(target[2*i] * img.shape[0]), int(target[2*i+1] *  img.shape[0])), 1, 1, (0, 0, 255), 2)
            #     cv2.circle(img, (int(target[2*i] *  img.shape[0]), int(target[2*i+1] *  img.shape[0])), 1, (0, 0, 255), 2)
            #     cv2.putText(img, str(i), (int(x_pred[i] * img.shape[0] ), int(y_pred[i] * img.shape[0])), 1, 1, (0, 0, 255), 2)
            #     cv2.circle(img, (int(x_pred[i] * img.shape[0]), int(y_pred[i] * img.shape[0])), 1, (0, 0, 255), 2)
            # cv2.imshow("x", img)
            # cv2.waitKey(0)
            # 还原
            x_s = x_pred
            y_s = y_pred

            x_loss += np.mean(np.abs(x_s - x_t))
            y_loss += np.mean(np.abs(y_s - y_t))
                # 还原图像
                # x_s = x_pred[i] * h * ratio[0]
                # y_s = y_pred[i] * w * ratio[1]
    total = len(image_ls)
    x_loss = x_loss / total
    y_loss = y_loss / total
    print("**************************   pytorch  *********************")
    print("pth x_loss:", x_loss)
    print("pth y_loss:", y_loss)

def get_labels(path):
    labels = {}
    images = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        img_name = line.split()[0]
        images.append(img_name)
        data = line.split()[1:]
        data = [float(x) for x in data]
        np_data = np.array(data)
        labels[img_name] = np_data

    return images, labels


if __name__ == '__main__':
    onnx_path = "../onnx/pipnet_mbv1_blurness_v5_64_retinaface_70ep.onnx"
    model_path = "../snapshots/Widerface/pipnetLD_mbv3_v8_64_fpn/epoch10.pth"
    flag = 0 # 2 3
    size = 64
    # label_txt = "/opt/data/face_landmark/val/WFLW_val_5.txt"
    label_txt = '/opt/data/face_landmark/face_keypoint_5/landmark/norm_benchmark_label.txt'
    # img_root = '/opt/data/face_landmark/val/'
    img_root = "/opt/data/face_landmark/face_keypoint_5/face_roi/"
    input_name = 'input'
    output_names = ['x_pred', 'y_pred', 'score']

    image_ls, labels = get_labels(label_txt)

    # onnx pred
    onnx_model = onnxruntime.InferenceSession(onnx_path, None)
    pipnet_onnx_pred(onnx_model, image_ls, root=img_root, targets=labels, size=(size,size))

    ## v3
    if flag == 3:
        mbnet = mobilenetv3_small()
        model = Pip_mbnetv3_small(mbnet=mbnet, input_size=size).cuda()
        model_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(model_dict)
        model.eval()

    # v2
    if flag == 2:
        mbnet = mobilenet_v2(width_mult=0.5,)
        model = Pip_mbnetv2_small(mbnet=mbnet).cuda()
        model_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(model_dict)
        model.eval()

    # v3 ld
    if flag == 4:
        mbnet = mobilenetv3_small()
        model = Pip_mbnetv3_small_ld(mbnet=mbnet, input_size=size).cuda()
        model_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(model_dict)
        model.eval()
    # pipnet_pt_pred(model, image_ls, img_root, labels, size=(size,size))