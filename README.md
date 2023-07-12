## Introduction
- 模型原本是基于pipnet,但是为了简化操作提高速度，于是采用了直接回归的方式，仍然保留名称pipnet
- 主要功能是对给出的图片进行判断是否是人脸，同时还会预测五个人脸特征点；
- 目前更新的版本增加了一个分类，预测图片是否模糊

## Installation
1. Install Python3 and PyTorch >= v1.1
2. Clone this repository.
```Shell
git clone https://github.com/Miracle-QCC/face_landmark_pred.git
```
3. Install the dependencies in requirements.txt.
```Shell
pip install -r requirements.txt
```

## Demo
```Shell
python train_5_points.py
```

test
```Shell
python tools/test_l1_loss.py
```

## Citation
````
@article{JLS21,
  title={Pixel-in-Pixel Net: Towards Efficient Facial Landmark Detection in the Wild},
  author={Haibo Jin and Shengcai Liao and Ling Shao},
  journal={International Journal of Computer Vision},
  publisher={Springer Science and Business Media LLC},
  ISSN={1573-1405},
  url={http://dx.doi.org/10.1007/s11263-021-01521-4},
  DOI={10.1007/s11263-021-01521-4},
  year={2021},
  month={Sep}
}
````

## Acknowledgement
We thank the following great works:
* [human-pose-estimation.pytorch](https://github.com/microsoft/human-pose-estimation.pytorch)
* [HRNet-Facial-Landmark-Detection](https://github.com/HRNet/HRNet-Facial-Landmark-Detection)
