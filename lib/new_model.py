'''MobileNetV3 in PyTorch.
See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch
from .FPN import FPN

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        expand_size = max(in_size // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(expand_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_size, in_size, kernel_size=1, bias=False),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, act, se, stride):
        super(Block, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = act(inplace=True)

        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = act(inplace=True)
        self.se = SeModule(expand_size) if se else nn.Identity()

        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.act3 = act(inplace=True)

        self.skip = None
        if stride == 1 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=in_size, kernel_size=3, groups=in_size, stride=2, padding=1,
                          bias=False),
                nn.BatchNorm2d(in_size),
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=True),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size == out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, groups=in_size, stride=2,
                          padding=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

    def forward(self, x):
        skip = x

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))

        if self.skip is not None:
            skip = self.skip(skip)
        return self.act3(out + skip)


class Bneck(nn.Module):
    def __init__(self, act=nn.Hardswish):
        super(Bneck, self).__init__()
        # self.bneck = nn.Sequential(
        #     Block(3, 16, 16, 16, nn.ReLU, True, 2),
        #     Block(3, 16, 72, 24, nn.ReLU, False, 2),
        #     Block(3, 24, 88, 24, nn.ReLU, False, 1),
        #     Block(5, 24, 96, 40, act, True, 2),
        #     Block(5, 40, 240, 40, act, True, 1),
        #     Block(5, 40, 240, 40, act, True, 1),
        #     Block(5, 40, 120, 48, act, True, 1),
        #     Block(5, 48, 144, 48, act, True, 1),
        #     Block(5, 48, 288, 96, act, True, 2),
        #     Block(5, 96, 576, 96, act, True, 1),
        #     Block(5, 96, 576, 96, act, True, 1),
        # )
        self.block1 = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU, True, 2),
                Block(3, 16, 72, 24, nn.ReLU, False, 2),
                Block(3, 24, 88, 24, nn.ReLU, False, 1),
                Block(5, 24, 96, 40, act, True, 1),
        )
        self.block2 = nn.Sequential(
            Block(5, 40, 240, 40, act, True, 2),
            Block(5, 40, 240, 40, act, True, 1),
            Block(5, 40, 120, 48, act, True, 1),
            Block(5, 48, 144, 48, act, True, 1),
        )

        # self.block3 = nn.Sequential(
        #         Block(5, 48, 288, 96, act, True, 2),
        #         Block(5, 96, 576, 96, act, True, 1),
        #         Block(5, 96, 576, 96, act, True, 1),
        # )
        self.init_params()
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        # x3 = self.block3(x2)

        return x1,x2


class MobileNetV3_Small_FPN(nn.Module):
    def __init__(self, num_classes=1000, act=nn.Hardswish):
        super(MobileNetV3_Small_FPN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = act(inplace=True)


        self.bneck = Bneck()

        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = act(inplace=True)
        self.fpn = FPN(out_channels=96)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out1,out2 = self.bneck(out)
        out = self.fpn(out1,out2)
        # out = torch.cat(out, 1)
        out = self.hs2(self.bn2(self.conv2(out)))
        return out


class MobileNetV3_Large(nn.Module):
    def __init__(self, num_classes=1000, act=nn.Hardswish):
        super(MobileNetV3_Large, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = act(inplace=True)

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU, False, 1),
            Block(3, 16, 64, 24, nn.ReLU, False, 2),
            Block(3, 24, 72, 24, nn.ReLU, False, 1),
            Block(5, 24, 72, 40, nn.ReLU, True, 2),
            Block(5, 40, 120, 40, nn.ReLU, True, 1),
            Block(5, 40, 120, 40, nn.ReLU, True, 1),
            Block(3, 40, 240, 80, act, False, 2),
            Block(3, 80, 200, 80, act, False, 1),
            Block(3, 80, 184, 80, act, False, 1),
            Block(3, 80, 184, 80, act, False, 1),
            Block(3, 80, 480, 112, act, True, 1),
            Block(3, 112, 672, 112, act, True, 1),
            Block(5, 112, 672, 160, act, True, 2),
            Block(5, 160, 672, 160, act, True, 1),
            Block(5, 160, 960, 160, act, True, 1),
        )

        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = act(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.linear3 = nn.Linear(960, 1280, bias=False)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = act(inplace=True)
        self.drop = nn.Dropout(0.2)

        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)

        out = self.hs2(self.bn2(self.conv2(out)))
        out = self.gap(out).flatten(1)
        out = self.drop(self.hs3(self.bn3(self.linear3(out))))

        return self.linear4(out)

class Pip_mbnetv3_small_fpn(nn.Module):
    def __init__(self, mbnet, num_lms=5, input_size=128):
        super(Pip_mbnetv3_small_fpn, self).__init__()
        self.num_lms = num_lms
        self.input_size = input_size
        self.features = mbnet

        if input_size == 128:
            self.ld_layer = nn.Conv2d(576, num_lms * 2, kernel_size=3, stride=2, padding=0)
            self.ld_point_layer = nn.Conv2d(num_lms * 2, out_channels=num_lms*2, kernel_size=7)

            self.score_layer = nn.Linear(576,1)
        elif input_size == 64:
            self.ld_layer = nn.Conv2d(576, num_lms * 2, kernel_size=3, stride=1, padding=0)
            self.ld_point_layer = nn.Conv2d(in_channels=num_lms * 2, out_channels=num_lms*2, kernel_size=6)
            # self.y_layer = nn.Conv2d(576, num_lms, kernel_size=1, stride=1, padding=0)
            # self.ld_liner = nn.Linear(16, 1)

            self.score_layer = nn.Linear(576, 1)

        # nn.init.normal_(self.cls_layer.weight, std=0.001)
        # if self.cls_layer.bias is not None:
        #     nn.init.constant_(self.cls_layer.bias, 0)

        nn.init.normal_(self.ld_layer.weight, std=0.001)
        if self.ld_layer.bias is not None:
            nn.init.constant_(self.ld_layer.bias, 0)


        nn.init.normal_(self.score_layer.weight, std=0.001)
        if self.score_layer.bias is not None:
            nn.init.constant_(self.score_layer.bias, 0)

        nn.init.normal_(self.ld_point_layer.weight, std=0.001)
        if self.ld_point_layer.bias is not None:
            nn.init.constant_(self.ld_point_layer.bias, 0)

    def forward(self, x):
        x = self.features(x)
        # x1 = self.cls_layer(x)
        xy = self.ld_layer(x)
        # x3 = self.y_layer(x)
        # if self.input_size == 128:
        #     score_x = F.avg_pool2d(x, 7)

        # elif self.input_size == 64:
        score_x = F.avg_pool2d(x, 8)

        xy = self.ld_point_layer(xy)
        xy = xy.view(-1,self.num_lms*2,1)
        score_x = score_x.view(score_x.size(0), -1)
        score = self.score_layer(score_x)
        x_pred = xy[:,::2,:]
        y_pred = xy[:,1::2,:]
        # if torch.onnx.is_in_onnx_export():
        #     print("顺序： x_pred, y_pred, score")
        #     # return x1,x2,x3,score
        #     return x_pred, y_pred, score
        return x_pred, y_pred, score

if __name__ == '__main__':
    net = MobileNetV3_Small_FPN()

    net.eval()
    x = torch.rand((1,3,64,64))

    net(x)