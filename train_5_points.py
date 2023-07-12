import datetime
import sys
sys.path.insert(0, '..')
import importlib

import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from lib.mobilenetv3 import mobilenetv3_small, mobilenetv3_small_light
from lib.mobilenetv3_fpn import MobileNetV3_Small_FPN
from lib.networks import *
import lib.data_utils as data_utils
from lib.functions import *
from tools.logging_tool import Logger
from lib.retinaface import RetinaFace
data_name = 'Widerface'
# TODO 使用了数据增强，并且增加了数据2004的widerface
save_name = 'pipnet_mbv1_v5_64_retinaface'
# backbone = sys.argv[1]
my_config = importlib.import_module('experiments.Widerface.pip_5_mbv3_l2_l1')
Config = getattr(my_config, 'Config')
cfg = Config()
cfg.data_name = data_name

os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu_id)

if not os.path.exists(os.path.join('./snapshots', cfg.data_name)):
    os.makedirs(os.path.join('./snapshots', cfg.data_name))
save_dir = os.path.join('./snapshots', cfg.data_name, save_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if not os.path.exists(os.path.join('./logs', cfg.data_name)):
    os.makedirs(os.path.join('./logs', cfg.data_name))

logger = Logger(logger=save_name, save_path=save_dir).getlog()

# logging.basicConfig(filename=os.path.join(log_dir, 'train.log'), level=logging.INFO)

# print('###########################################')
# print('experiment_name:', cfg.experiment_name)
# print('data_name:', cfg.data_name)
# print('det_head:', cfg.det_head)
# print('net_stride:', cfg.net_stride)
# print('batch_size:', cfg.batch_size)
# print('init_lr:', cfg.init_lr)
# print('num_epochs:', cfg.num_epochs)
# print('decay_steps:', cfg.decay_steps)
# print('input_size:', cfg.input_size)
# print('backbone:', cfg.backbone)
# print('pretrained:', cfg.pretrained)
# print('criterion_cls:', cfg.criterion_cls)
# print('criterion_reg:', cfg.criterion_reg)
# print('cls_loss_weight:', cfg.cls_loss_weight)
# print('reg_loss_weight:', cfg.reg_loss_weight)
# print('num_lms:', cfg.num_lms)
# print('save_interval:', cfg.save_interval)
# # print('num_nb:', cfg.num_nb)
# print('use_gpu:', cfg.use_gpu)
# print('gpu_id:', cfg.gpu_id)
# print('###########################################')
logger.info('###########################################')
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
# logging.info('num_nb: {}'.format(cfg.num_nb))
logger.info('use_gpu: {}'.format(cfg.use_gpu))
logger.info('gpu_id: {}'.format(cfg.gpu_id))
logger.info('###########################################')

# if cfg.det_head == 'pip':
#     if cfg.backbone == 'mobilenet_v3_small':
#         # mobilnetv3 = mobilenet_v3_small()
#         net = Pip_mbnetv3_small(num_lms=cfg.num_lms, input_size=cfg.input_size)
#
#     else:
#         print('No such backbone!')
#         exit(0)
# else:
#     print('No such head:', cfg.det_head)
#     exit(0)

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

points_flip = None

if cfg.data_name == "Widerface":
    points_flip = [2, 1, 3, 5, 4] # five landmarks
    points_flip = (np.array(points_flip) - 1).tolist()
else:
    print('No such data!')
    exit(0)

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])



def load_model(model, checkpoint):
    model_CKPT = torch.load(checkpoint)
    model_dict = model.state_dict()
    pretrained_dict = model_CKPT
    # 将不在model中的参数过滤掉
    new_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict.keys()}
    # new_dict.pop("classifier.weight")
    # new_dict.pop("classifier.bias")
    model_dict.update(new_dict)
    model.load_state_dict(model_dict)

    return model

if __name__ == '__main__':

    # TODO 测试L1 loss
    root = '/opt/data/face_landmark/train'
    labels = get_label_5(root, 'norm_label_v5_6_14.txt')

    # root = '/opt/data/face_landmark/train'
    # labels = get_label_5(root, 'norm_label_v2.txt')



    if cfg.det_head == 'pip':
        train_data = data_utils.ImageFolder_pip_5(root,
                                                  labels, cfg.input_size, cfg.num_lms,
                                                  cfg.net_stride, points_flip,
                                                  transforms.Compose([
                                                  # transforms.RandomVerticalFlip(p=0.5),
                                                  transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
                                                  transforms.ToTensor(),
                                                  normalize]))
    else:
        print('No such head:', cfg.det_head)
        exit(0)

    # TODO  light v3   128 -> 64    model - > ld model
    # mbnet = mobilenetv3_small()
    input_size = 64
    # mbnet = mobilenetv3_small()
    # mbnet = MobileNetV3_Small_FPN()
    # mbnet = load_model(mbnet, '300_act3_mobilenetv3_small.pth')
    # net = Pip_mbnetv3_small(mbnet=mbnet, num_lms=cfg.num_lms, input_size=input_size)
    # net = Pip_mbnetv3_small_ld(mbnet, num_lms=5, input_size=64)
    # net = Pip_mbnetv3_small_fpn(mbnet=mbnet, num_lms=5, input_size=64)


    # TODO retinaface
    cfg_mnet = {
        'name': 'mobilenet0.25',
        'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
        'in_channel': 32,
        'out_channel': 64
    }
    net = RetinaFace(cfg=cfg_mnet)
    net = net.to(device)



    # model_path = 'snapshots/Widerface/pipnet_mbv3_v2/epoch100.pth'
    # mbnet = mobilenetv3_small()
    # mbnet = mobilenetv3_small_light()
    # model = Pip_mbnetv3_small(mbnet=mbnet)
    # model_dict = torch.load(model_path, map_location='cpu')
    # model.load_state_dict(model_dict)
    # net = model.to(device)
    # if cfg.pretrained:
    #     optimizer = optim.Adam(net.parameters(), lr=cfg.init_lr)
    # else:
    #

    optimizer = optim.Adam(net.parameters(), lr=cfg.init_lr, weight_decay=5e-5)
    # optimizer = optim.SGD(net.parameters(), lr=cfg.init_lr, weight_decay=5e-5,momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.decay_steps, gamma=0.1)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    train_model_5(cfg.det_head, net, train_loader, criterion_cls, criterion_reg, cfg.cls_loss_weight, cfg.reg_loss_weight, optimizer, cfg.num_epochs, scheduler, save_dir, cfg.save_interval, device, logger)

