"""
测试人脸特征点的指标，如果预测值里面有特征点，那么会计算特征点的瞳距是否大于20

"""
import numpy as np
from collections import Counter

import onnxruntime
from sklearn.metrics import f1_score, accuracy_score, roc_curve, precision_score, recall_score, roc_auc_score, log_loss
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from val_map import pred_imgs_onnx,get_imgs
'''精确率'''
def get_precision(y, y_pre):
    '''
    :param y: array，真实值
    :param y_pre: array，预测值
    :return: float
    '''
    return precision_score(y, y_pre)

'''召回率'''
def get_recall(y, y_pre):
    '''
    :param y: array，真实值
    :param y_pre: array，预测值
    :return: float
    '''
    return recall_score(y, y_pre)

'''F1'''
def get_f1(y, y_pre):
    '''
    :param y: array，真实值
    :param y_pre: array，预测值
    :return: float
    '''
    return f1_score(y, y_pre)

def get_data(path):
    mp = {}
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        key = line.split()[0]
        data = line.split()[1:]
        # print(line)
        mp[key] = [float(x) for x in data]
    return mp

'''AUC'''
def get_auc(y, y_score):
    '''
    :param y: array，真实值
    :param y_score: array，预测概率值
    :return: float
    '''
    return roc_auc_score(y, y_score)
def get_failcase(y_t, y_pred, keys, scores):
    n = len(y_t)
    failcase = []
    cnt = Counter(y_pred)
    Total_p = cnt[1]
    Total_n = cnt[0]
    tp = 0
    fn = 0
    for i in tqdm(range(n)):
        if y_t[i] != y_pred[i]:
            failcase.append(keys[i])
        else:
            if y_t[i] == 1:
                tp += 1
            else:
                fn += 1

    print("正样本准确率：{}/{}={:.2f}%".format(tp, Total_p, tp / Total_p * 100))
    print("负样本准确率：{}/{}={:.2f}%".format(fn, Total_n, fn / Total_n * 100))
    preces = 1 - len(failcase) / n
    return failcase, preces
if __name__ == '__main__':
    # gt 保存的是
    """
    pos / wflw_test_0101.jpg 1
    pos / 736_4.jpg 1
    blur / 0
    _Parade_marchingband_1_291_00000000.jpg 0
    blur / 0
    _Parade_marchingband_1_6_00000000.jpg 0
   """
    gt_path = "/home/qcj/workcode/CVIAI_TOOL/TXT/eval_ipc_labels.txt"
    onnx_path = "../onnx/pipnet_mbv1_blurness_v5_64_retinafaceBN_50ep.onnx"
    onnx_model = onnxruntime.InferenceSession(onnx_path, None)
    input_name = 'input'
    output_names = ['x_pred', 'y_pred', 'score']

    img_path = '/opt/data/face_landmark/eval_data/eval_ipc_labels.txt'
    # img_path = '/opt/data/face_landmark/ipc_6_14_labels.txt'
    img_root = '/opt/data/face_landmark/eval_data'  # '/opt/data/face_landmark/capture_pic_6_14'
    # img_root = '/opt/data/face_landmark/capture_pic_6_14'
    imgs = get_imgs(img_path)
    pred_mp = pred_imgs_onnx(imgs, img_root, onnx_model, "../TXT/eval_retinaface_blurness_100ep.txt", size=(64, 64))
    gt_mp = get_data(gt_path)

    labels = []
    preds = []
    keys = []
    for key in pred_mp:
        labels.append(gt_mp[key])
        preds.append(pred_mp[key])
        keys.append(key)
    labels = [int(x[0]) for x in labels]
    confs = [0.1,0.2,0.3,0.4,0.5]
    for conf in confs:
        pred_label = []
        for x in preds:
            if len(x) > 1:
                if x[0] < conf or abs(x[1] - x[3]) < 0.156:
                    pred_label.append(0)
                else:
                    pred_label.append(1)
            else:
                if x[0] < conf:
                    pred_label.append(0)
                else:
                    pred_label.append(1)

        # preces = get_precision(labels, pred_label)
        # auc = get_auc(labels, preds)
        # precision, recall, thresholds = precision_recall_curve(labels, preds)
        # print(precision)
        # print(recall)
        # print(thresholds)

        # plt.plot(precision, recall)
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.show()
        print("阈值：",conf)
        failces, preces = get_failcase(labels, pred_label, keys, preds)
        # f = open("TXT/failcase_benchmark.txt_{}".format(conf), 'w')
        for key in failces:
            data = pred_mp[key]
            data = [str(x) for x in data]

        # print((failces))
        print("精确率：{:.5f}%".format(preces * 100))
        # print(thresholds)
        # print("AUC:", auc)