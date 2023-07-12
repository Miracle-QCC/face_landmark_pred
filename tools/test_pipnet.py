import torch
from lib.networks import Pip_mbnetv3_small
from lib.mobilenetv3 import mobilenetv3_small
from PIL import Image
import torchvision.transforms as transforms
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
import numpy as np
import cv2
transformer = transforms.Compose([transforms.ToTensor(),
                  normalize])


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

def pred(model, img_tensor, stride = 32):
    with torch.no_grad():
        cls,x,y,score = model(img_tensor)
        cls = cls.squeeze()
        x = x.squeeze()
        y = y.squeeze()
        score = 1 / (1+torch.exp(-score))
        landmarks = []
        if score < 0.6:
            print("非人脸")
            return
        for i in range(5):
            id_i, id_j = getMaxPos(cls[i])
            x_b = stride * id_i
            y_b = stride * id_j

            x_offset = x[i,id_i,id_j] * stride
            y_offset = y[i,id_i,id_j] * stride

            x_ = x_b + x_offset
            y_ = y_b + y_offset
            landmarks.append(x_)
            landmarks.append(y_)
    return landmarks




if __name__ == '__main__':
    model_path = 'snapshots/Widerface/pipnet_mbv3_pretrain_heatmap/epoch100.pth'
    mbnet = mobilenetv3_small()
    model = Pip_mbnetv3_small(mbnet=mbnet)
    model_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(model_dict)
    img_path = '/opt/data/face_landmark/train/WFLW/wflw_train_0001.jpg'
    img = Image.open(img_path).convert('RGB')
    img = letter_bbox(img)
    tensor = transformer(img)

    landmarks = pred(model, tensor.unsqueeze(0))
    img_show = np.array(img)
    for i in range(5):
        cv2.putText(img_show, str(i), (int(landmarks[2*i]), int(landmarks[2*i+1])), 1, 1, (0, 0, 255), 2)

    cv2.imshow("x", img_show)
    cv2.waitKey(0)