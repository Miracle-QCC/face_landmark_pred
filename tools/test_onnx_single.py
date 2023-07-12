import os

import onnxruntime
import onnx
import cv2
import numpy as np
from tqdm import tqdm


def pipnet_onnx_pred(onnx_model, img_path, target_size=(128,128)):
    img = cv2.imread(img_path)
    img_processed = cv2.resize(img,target_size)
    blob = cv2.dnn.blobFromImage(img_processed, 1.0 / 127.5, target_size, (127.5, 127.5, 127.5), swapRB=True)
    net_outs = onnx_model.run(output_names, {input_name: blob})
    x_pred, y_pred, score = net_outs
    score = 1 / (1 + np.exp(-score))
    x_pred = x_pred[0, :, 0]
    y_pred = y_pred[0, :, 0]
    score = float(score)
    print(score)
    if score < 0.3:
        print("非人脸")
    h,w = img.shape[:2]
    for i in range(5):
        # cv2.putText(img, str(i),
        #             (int(x_pred[i] * w), int(y_pred[i] * h)), 1, 1,
        #             (0, 0, 255), 2)
        cv2.circle(img, (int(x_pred[i] * w), int(y_pred[i] * h)),
                   1, (0, 0, 255), 2)
    cv2.imshow("x", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    # onnx pred
    onnx_path = "../onnx/pipnet_mbv3_64_v5.onnx"
    input_name = 'input'
    output_names = ['x_pred', 'y_pred', 'score']
    img_path = "/home/qcj/cvitek/test_00000044-face1.png"

    onnx_model = onnxruntime.InferenceSession(onnx_path, None)
    pipnet_onnx_pred(onnx_model, img_path,target_size=(64,64))

