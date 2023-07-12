import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict

from thop import profile

from .net import MobileNetV1 as MobileNetV1
from .net import FPN as FPN
from .net import SSH as SSH
from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.utils.state import enable_calibration, enable_quantization
from mqbench.convert_deploy import convert_deploy

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class ClassHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 10)


class RetinaFace(nn.Module):
    def __init__(self, cfg=None, phase='train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace, self).__init__()
        self.phase = phase
        backbone = None
        backbone = MobileNetV1()

        # if cfg['name'] == 'mobilenet0.25':
        #     backbone = MobileNetV1()
        #     if cfg['pretrain']:
        checkpoint = torch.load("/home/qcj/workcode/PIPNet/weights/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]  # remove module.
            new_state_dict[name] = v
        # load params
        backbone.load_state_dict(new_state_dict)
        # elif cfg['name'] == 'Resnet50':
        #     import torchvision.models as models
        #     backbone = models.resnet50(pretrained=cfg['pretrain'])

        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)

        self.ClassFeature = nn.Conv2d(kernel_size=3, in_channels=64, out_channels=64, stride=2)
        self.Class = nn.Conv2d(kernel_size=3, in_channels=64, out_channels=1)

        self.LandmarkFeature = nn.Conv2d(kernel_size=5, in_channels=64, out_channels=64, stride=1)
        self.Landmark = nn.Conv2d(kernel_size=4, in_channels=64, out_channels=10)

    def forward(self, inputs):
        out = self.body(inputs)


        # FPN
        fpn = self.fpn(out)


        # SSH
        feature = self.ssh1(fpn[0])

        cls_feature = self.ClassFeature(feature)
        cls = self.Class(cls_feature)

        ld_feature = self.LandmarkFeature(feature)
        ld = self.Landmark(ld_feature)


        ld = ld.view(-1,10)
        x_pred = ld[:,::2]
        y_pred = ld[:,1::2]

        return x_pred,y_pred,cls


class RetinaFace_Blurness(nn.Module):
    def __init__(self, cfg=None, phase='train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace_Blurness, self).__init__()
        self.phase = phase
        backbone = None
        backbone = MobileNetV1()

        checkpoint = torch.load("/home/qcj/workcode/PIPNet/weights/mobilenetV1X0.25_pretrain.tar",
                                map_location=torch.device('cpu'))
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]  # remove module.
            new_state_dict[name] = v
        # load params
        backbone.load_state_dict(new_state_dict)
        # elif cfg['name'] == 'Resnet50':
        #     import torchvision.models as models
        #     backbone = models.resnet50(pretrained=cfg['pretrain'])

        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)

        self.Feature_Ext = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=1,stride=1)
        # self.Feature_BN = nn.BatchNorm2d(64)

        self.ClassFeature = nn.Conv2d(kernel_size=3, in_channels=64, out_channels=64, stride=2)
        self.Class = nn.Conv2d(kernel_size=3, in_channels=64, out_channels=1)

        self.LandmarkFeature = nn.Conv2d(kernel_size=5, in_channels=64, out_channels=64, stride=1)
        self.Landmark = nn.Conv2d(kernel_size=4, in_channels=64, out_channels=10)

        # blur pred
        self.BlurFeature = nn.Conv2d(kernel_size=3, in_channels=64, out_channels=64, stride=2)
        self.BlurScore = nn.Conv2d(kernel_size=3, in_channels=64, out_channels=1)
        self.hs = h_swish()
    def forward(self, inputs):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)


        # SSH
        feature = self.ssh1(fpn[0])
        feature = self.hs(self.Feature_Ext(feature))
        # feature = self.hs(self.Feature_BN(self.Feature_Ext(feature)))

        cls_feature = self.ClassFeature(feature)
        cls = self.Class(cls_feature)

        ld_feature = self.LandmarkFeature(feature)
        ld = self.Landmark(ld_feature)

        ld = ld.view(-1, 10)
        x_pred = ld[:, ::2]
        y_pred = ld[:, 1::2]

        # blur
        blur_feature = self.BlurFeature(feature)
        blur_score = self.BlurScore(blur_feature)

        if torch.onnx.is_in_onnx_export():
            score = torch.cat([cls,blur_score], dim=1)
            return x_pred,y_pred,score

        return x_pred, y_pred, cls, blur_score


if __name__ == '__main__':
    cfg_mnet = {
        'name': 'mobilenet0.25',
        'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
        'in_channel': 32,
        'out_channel': 64
    }
    net = RetinaFace_Blurness(cfg=cfg_mnet)
    # net.eval()
    x = torch.rand((1, 3, 64, 64))
    net.eval()
    net(x)
    flops, params = profile(net, (x,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
    # backend = BackendType.Sophgo_TPU
    # net = prepare_by_platform(net, backend)
    # net.eval()
    #
    # enable_calibration(net)
    # x_red,y_pred,score = net(x)
    # # logger.info("QAT test start")
    # # for i, data in enumerate(train_loader):
    # #     img_names, inputs, target, label = data
    # #     inputs = inputs.to(device)
    # #     net(inputs)
    # #     break
    # # logger.info("QAT test done!")
    # net.train()
    # enable_quantization(net)

    # for epoch in range(num_epochs):
    #     logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
    #     logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # logger.info('-' * 10)
        # logging.info('-' * 10)
        # net.train()
        # epoch_loss = 0.0