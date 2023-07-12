import os

import onnxruntime
import onnx
import cv2
import numpy as np
from tqdm import tqdm


def pipnet_onnx_pred(onnx_model, img_path, target_size=(128,128)):
    img = cv2.imread(img_path.split()[0])
    img_name = img_path.split("/")[-1][:-1]
    try:
        img_processed = cv2.resize(img,target_size)
    except:
        return
    blob = cv2.dnn.blobFromImage(img_processed, 1.0 / 127.5, target_size, (127.5, 127.5, 127.5), swapRB=True)
    net_outs = onnx_model.run(output_names, {input_name: blob})
    x_pred, y_pred, score = net_outs
    score = np.squeeze(score)
    score = 1 / (1 + np.exp(-score))
    blur_score = None
    if len(score) == 2:
        blur_score = score[1]
        blur_score = "{:.4f}".format(float(blur_score))
        score = score[0]

    if len(x_pred.shape) == 3:
        x_pred = x_pred[0, :, 0]
        y_pred = y_pred[0, :, 0]
    elif len(x_pred.shape) == 2:
        x_pred = x_pred[0, :]
        y_pred = y_pred[0, :]
    h, w = img.shape[:2]
    for i in range(5):
        # cv2.putText(img, str(i),
        #             (int(x_pred[i] * w), int(y_pred[i] * h)), 1, 1,
        #             (0, 0, 255), 2)
        cv2.circle(img, (int(x_pred[i] * w), int(y_pred[i] * h)),
                   1, (0, 0, 255), 2)
    # cv2.imshow("x", img_processed)
    # cv2.waitKey(0)
    # cv2.imwrite("../images/eval_imgs/" + img_name, img)
    score = "{:.4f}".format(float(score))
    # cv2.putText(img, str(score),
    #                 (int(0), int(0)), 1, 1,
    #                 (0, 0, 255), 2)
    score = float(score)
    if score < 0.5:
        # print("非人脸")

        # x_pred = x_pred[0,:,0]
        # y_pred = y_pred[0,:,0]
        # h,w = img.shape[:2]
        # for i in range(5):
        #     cv2.putText(img, str(i),
        #                 (int(x_pred[i] * w), int(y_pred[i] * h)), 1, 1,
        #                 (0, 0, 255), 2)
        #     cv2.circle(img, (int(x_pred[i] * w), int(y_pred[i] * h)),
        #                1, (0, 0, 255), 2)
        # cv2.imshow("x", img_processed)
        # cv2.waitKey(0)
        if blur_score:
            cv2.putText(img, blur_score,
                        (int(w // 2), int(h // 2)), 1, 1,
                        (0, 0, 255), 2)
        cv2.imwrite("../images/neg_imgs/" + img_name, img)
    else:
        if blur_score:
            cv2.putText(img, blur_score,
                    (int(w // 2), int(h // 2)), 1, 1,
                    (0, 0, 255), 2)
        cv2.imwrite("../images/pos_imgs/" + img_name, img)


if __name__ == '__main__':
    # onnx pred
    onnx_path = "../onnx/pipnet_mbv1_blurness_v5_64_retinaface_50ep.onnx"
    input_name = 'input'
    output_names = ['x_pred', 'y_pred', 'score']
    # img_path = "/opt/data/face_landmark/train/side/SZ6462_1685797511_3460045_0.jpg"
    with open("/home/qcj/workcode/CVIAI_TOOL/TXT/blur_imgs.txt", 'r') as f:
        imgs = f.readlines()

    if not os.path.exists("../images/neg_imgs"):

        os.makedirs("../images/neg_imgs")
    if not os.path.exists("../images/pos_imgs"):
        os.makedirs("../images/pos_imgs")

    onnx_model = onnxruntime.InferenceSession(onnx_path, None)
    for img_p in tqdm(imgs):
        pipnet_onnx_pred(onnx_model, img_p, target_size=(64,64))

