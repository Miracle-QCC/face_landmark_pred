import datetime

import sys
sys.path.insert(0, '..')
import importlib

import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from lib.mobilenetv3 import mobilenetv3_small,mobilenetv3_small_light
from lib.mobilenetv3_fpn import MobileNetV3_Small_FPN
from lib.networks import *
from lib.retinaface import RetinaFace
import lib.data_utils as data_utils
from lib.functions import *

from logging_tool import Logger

from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.utils.state import enable_calibration, enable_quantization
from mqbench.convert_deploy import convert_deploy

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])





def load_model(model, checkpoint):
    model_CKPT = torch.load(checkpoint)
    model_dict = model.state_dict()
    pretrained_dict = model_CKPT
    # 将不在model中的参数过滤掉
    new_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict.keys()}
    model_dict.update(new_dict)
    model.load_state_dict(model_dict)

    return model

def train_model_5_qat(det_head, net, train_loader, criterion_cls, criterion_reg, cls_loss_weight, reg_loss_weight, optimizer, num_epochs, scheduler, save_dir, save_interval, device):
    backend = BackendType.Sophgo_TPU
    net = prepare_by_platform(net, backend).to(device)
    net.eval()

    enable_calibration(net)
    logger.info("QAT test start")
    for i, data in enumerate(train_loader):
        img_names, inputs, target, label = data
        inputs = inputs.to(device)
        net(inputs)
        break
    logger.info("QAT test done!")
    net.train()
    enable_quantization(net)
    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logger.info('-' * 10)
        # logging.info('-' * 10)
        net.train()
        epoch_loss = 0.0


        for i, data in enumerate(train_loader):

            if det_head == 'pip':
                img_names, inputs, target, label = data
                inputs = inputs.to(device)
                target = target.to(device)
                x_label = target[:,::2]
                y_label = target[:,1::2]
                label = label.to(device).float()
                x_pred, y_pred, score = net(inputs)
                loss_x, loss_y, loss_score = compute_loss_pip_5_(x_pred, y_pred, score, x_label, y_label, label, criterion_reg)
                loss = reg_loss_weight*loss_x + reg_loss_weight*loss_y + loss_score * cls_loss_weight
            else:
                logger.info('No such head:', det_head)
                exit(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i%10 == 0:
                if det_head == 'pip':
                    logger.info('[Epoch {:d}/{:d}, Batch {:d}/{:d}] <Total loss: {:.6f}> <x loss: {:.6f}> <y loss: {:.6f}> <score loss: {:.6f}> '.format(
                        epoch, num_epochs-1, i, len(train_loader)-1, loss.item(), reg_loss_weight*loss_x.item(), reg_loss_weight*loss_y.item(), reg_loss_weight*loss_score.item()))
                else:
                    logger.info('No such head:', det_head)
                    exit(0)
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        input_shape = {'data': [1, 3,64, 64]}
        net.eval()
        # print(self.model)
        convert_deploy(net, backend, input_shape, output_path=str("../onnx"), model_name=save_name)
    return net



if __name__ == '__main__':

    config_path = 'experiments.Widerface.pip_5_mbv3_l2_l1'
    data_name = 'Widerface'

    my_config = importlib.import_module(config_path)
    Config = getattr(my_config, 'Config')
    cfg = Config()
    cfg.experiment_name = 'pip_5_mbv3.py'
    cfg.data_name = data_name
    save_name='pipnet_mbv1_qat_retinaface_100ep'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu_id)
    # save_name = 'pipnet_mbv3_light_qat'
    if not os.path.exists(os.path.join('./snapshots', cfg.data_name)):
        os.makedirs(os.path.join('./snapshots', cfg.data_name))
    save_dir = os.path.join('./snapshots', cfg.data_name, save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    root = '/opt/data/face_landmark/train'
    labels = get_label_5(root, 'norm_label_v8_pxpand.txt')

    logger = Logger(logger="testlogger", save_path=save_dir).getlog()

    logger.info('###########################################')
    logger.info('experiment_name: {}'.format(cfg.experiment_name))
    logger.info('data_name: {}'.format(cfg.data_name))
    logger.info('det_head: {}'.format(cfg.det_head))
    logger.info('net_stride: {}'.format(cfg.net_stride))
    logger.info('batch_size: {}'.format(cfg.batch_size))
    logger.info('init_lr: {}'.format(cfg.init_lr))
    logger.info('num_epochs: {}'.format(cfg.num_epochs))
    logger.info('decay_steps: {}'.format(cfg.decay_steps))
    logger.info('input_size: {}'.format(cfg.input_size))
    logger.info('backbone: {}'.format(cfg.backbone))
    logger.info('pretrained: {}'.format(cfg.pretrained))
    logger.info('criterion_cls: {}'.format(cfg.criterion_cls))
    logger.info('criterion_reg: {}'.format(cfg.criterion_reg))
    logger.info('cls_loss_weight: {}'.format(cfg.cls_loss_weight))
    logger.info('reg_loss_weight: {}'.format(cfg.reg_loss_weight))
    logger.info('num_lms: {}'.format(cfg.num_lms))
    logger.info('save_interval: {}'.format(cfg.save_interval))
    logger.info('use_gpu: {}'.format(cfg.use_gpu))
    logger.info('gpu_id: {}'.format(cfg.gpu_id))
    logger.info('###########################################')

    if cfg.use_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    criterion_cls = None
    if cfg.criterion_cls == 'l2':
        criterion_cls = nn.MSELoss()
    elif cfg.criterion_cls == 'l1':
        criterion_cls = nn.L1Loss()
    else:
        print('No such cls criterion:', cfg.criterion_cls)

    criterion_reg = None
    if cfg.criterion_reg == 'l1':
        criterion_reg = nn.L1Loss()
    elif cfg.criterion_reg == 'l2':
        criterion_reg = nn.MSELoss()
    else:
        print('No such reg criterion:', cfg.criterion_reg)

    if cfg.data_name == "Widerface":
        points_flip = [2, 1, 3, 5, 4]  # five landmarks
        points_flip = (np.array(points_flip) - 1).tolist()
    else:
        print('No such data!')
        exit(0)

    if cfg.det_head == 'pip':
        train_data = data_utils.ImageFolder_pip_5(root,
                                                  labels, cfg.input_size, cfg.num_lms,
                                                  cfg.net_stride, points_flip,
                                                  transforms.Compose([
                                                  transforms.ToTensor(),
                                                  normalize]))
    else:
        print('No such head:', cfg.det_head)
        exit(0)


    # mbnet = mobilenetv3_small()
    # mbnet = MobileNetV3_Small_FPN()
    # net = Pip_mbnetv3_small(mbnet=mbnet, num_lms=cfg.num_lms, input_size=cfg.input_size)

    model_path = '../snapshots/Widerface/pipnetLD_mbv3_v8_64_retinaface/epoch100.pth'
    QAT_epochs = 3
    # mbnet = mobilenetv3_small()
    model = cfg_mnet = {
        'name': 'mobilenet0.25',
        'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
        'in_channel': 32,
        'out_channel': 64
    }
    model = RetinaFace(cfg=cfg_mnet)
    # model = Pip_mbnetv3_small_fpn(mbnet=mbnet, input_size=64)
    model_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(model_dict)
    net = model.to(device)

    optimizer = optim.Adam(net.parameters(), lr=1e-5, weight_decay=0)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.decay_steps, gamma=0.1)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    train_model_5_qat(cfg.det_head, net, train_loader, criterion_cls, criterion_reg, cfg.cls_loss_weight, cfg.reg_loss_weight, optimizer, QAT_epochs, scheduler, save_dir, cfg.save_interval, device)

