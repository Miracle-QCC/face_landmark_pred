import torch
import torch.nn as nn
import math

from thop import profile
from torch.nn import init
from FPN import FPN

__all__ = ['mobilenetv2']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2_FPN(nn.Module):
    def __init__(self, width_mult=0.25):
        super(MobileNetV2_FPN, self).__init__()
        self.cfgs1 = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],

        ]
        self.cfgs2 = [
            [6, 32, 3, 2],
            [6, 64, 4, 1],
            [6, 96, 3, 1],
        ]
        self.cfgs3 = [
            [6, 160, 3, 2],
            [6, 320, 1, 1]]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers1 = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs1:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers1.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features1 = nn.Sequential(*layers1)

        layers2 = []
        for t, c, n, s in self.cfgs2:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers2.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features2 = nn.Sequential(*layers2)

        layers3 = []
        for t, c, n, s in self.cfgs3:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers3.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features3 = nn.Sequential(*layers3)
        self.fpn = FPN(out_channels=96)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x1)
        x3 = self.features3(x2)

        out = self.fpn(input=[x2,x3])
        return out



def load_model(model, checkpoint):
    model_CKPT = torch.load(checkpoint)
    model_dict = model.state_dict()
    pretrxained_dict = model_CKPT
    # 将不在model中的参数过滤掉
    new_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict.keys()}
    # new_dict.pop("classifier.weight")
    # new_dict.pop("classifier.bias")
    model_dict.update(new_dict)
    model.load_state_dict(model_dict)

    return model

def mobilenet_v2(pretrained_path=None, width_mult=1.):
    model = MobileNetV2_FPN(width_mult=width_mult)

    if pretrained_path:
        # try:
        #     from torch.hub import load_state_dict_from_url
        # except ImportError:
        #     from torch.utils.model_zoo import load_url as load_state_dict_from_url
        # state_dict = load_state_dict_from_url(
        #     'https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv2_1.0-f2a8633.pth.tar?dl=1', progress=True)
        # model.load_state_dict(state_dict)
        model = load_model(model, pretrained_path)
    return model


if __name__ == '__main__':
    net = mobilenet_v2(pretrained_path=None, width_mult=0.5)
    net.eval()
    x = torch.randn((1,3,64,64))
    # net(x)
    flops, params = profile(net, (x,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))



