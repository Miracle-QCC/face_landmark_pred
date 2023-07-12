import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from thop import profile
from torch.nn import init
# from mobilenetv3 import mobilenetv3_small_light
# net_stride output_size
# 128        2x2
# 64         4x4
# 32         8x8
# pip regression, resnet101
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
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

def conv_bn(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )

def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),
    )


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class FPN(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(FPN,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride = 1, leaky = leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride = 1, leaky = leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride = 1, leaky = leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky = leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky = leaky)

    def forward(self, input):
        # names = list(input.keys())
        input = list(input.values())

        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out

class Pip_mbnetv3_small(nn.Module):
    def __init__(self, mbnet, num_lms=5, input_size=128):
        super(Pip_mbnetv3_small, self).__init__()
        self.num_lms = num_lms
        self.input_size = input_size
        self.features = mbnet

        if input_size == 128:
        # self.cls_layer = nn.Conv2d(576, num_lms, kernel_size=1, stride=1, padding=0)
            self.x_layer = nn.Conv2d(576, num_lms, kernel_size=1, stride=1, padding=0)
            self.y_layer = nn.Conv2d(576, num_lms, kernel_size=1, stride=1, padding=0)
            self.x_liner = nn.Linear(16,1)
            self.y_liner = nn.Linear(16,1)

            self.score_layer = nn.Linear(576,1)
        elif input_size == 64:
            self.x_layer = nn.Conv2d(576, num_lms, kernel_size=1, stride=1, padding=0)
            self.y_layer = nn.Conv2d(576, num_lms, kernel_size=1, stride=1, padding=0)
            self.x_liner = nn.Linear(4, 1)
            self.y_liner = nn.Linear(4, 1)

            self.score_layer = nn.Linear(576, 1)

        # nn.init.normal_(self.cls_layer.weight, std=0.001)
        # if self.cls_layer.bias is not None:
        #     nn.init.constant_(self.cls_layer.bias, 0)

        nn.init.normal_(self.x_layer.weight, std=0.001)
        if self.x_layer.bias is not None:
            nn.init.constant_(self.x_layer.bias, 0)

        nn.init.normal_(self.y_layer.weight, std=0.001)
        if self.y_layer.bias is not None:
            nn.init.constant_(self.y_layer.bias, 0)

        nn.init.normal_(self.score_layer.weight, std=0.001)
        if self.score_layer.bias is not None:
            nn.init.constant_(self.score_layer.bias, 0)

        nn.init.normal_(self.x_liner.weight, std=0.001)
        if self.x_liner.bias is not None:
            nn.init.constant_(self.x_liner.bias, 0)

        nn.init.normal_(self.y_liner.weight, std=0.001)
        if self.y_liner.bias is not None:
            nn.init.constant_(self.y_liner.bias, 0)

    def forward(self, x):
        x = self.features(x)
        # x1 = self.cls_layer(x)
        x2 = self.x_layer(x)
        x3 = self.y_layer(x)
        if self.input_size == 128:
            x2 = x2.view(-1,5,16)
            x3 = x3.view(-1,5,16)

            x2 = self.x_liner(x2)
            x3 = self.y_liner(x3)
            # x2 = F.avg_pool2d(x2, 4)
            # x3 = F.avg_pool2d(x3, 4)
            score_x = F.avg_pool2d(x, 4)
        elif self.input_size == 64:
            x2 = x2.view(-1, 5, 4)
            x3 = x3.view(-1, 5, 4)

            x2 = self.x_liner(x2)
            x3 = self.y_liner(x3)
            # x2 = F.avg_pool2d(x2, 4)
            # x3 = F.avg_pool2d(x3, 4)
            score_x = F.avg_pool2d(x, 2)
        score_x = score_x.view(score_x.size(0), -1)
        score = self.score_layer(score_x)

        if torch.onnx.is_in_onnx_export():
            print("顺序： x  ,y  score")
            # return x1,x2,x3,score
            return x2,x3,score
        # return x1, x2, x3, score
        return x2,x3,score


class Pip_mbnetv3_small_ld(nn.Module):
    def __init__(self, mbnet, num_lms=5, input_size=128):
        super(Pip_mbnetv3_small_ld, self).__init__()
        self.num_lms = num_lms
        self.input_size = input_size
        self.features = mbnet

        if input_size == 128:
        # self.cls_layer = nn.Conv2d(576, num_lms, kernel_size=1, stride=1, padding=0)
            self.ld_layer = nn.Conv2d(576, num_lms * 2, kernel_size=1, stride=1, padding=1)
            self.ld_point_layer = nn.Conv2d(num_lms * 2, 1, kernel_size=3)
            # self.y_layer = nn.Conv2d(576, num_lms, kernel_size=1, stride=1, padding=0)
            # self.y_liner = nn.Linear(16,1)

            self.score_layer = nn.Linear(576,1)
        elif input_size == 64:
            self.ld_layer = nn.Conv2d(576, num_lms * 2, kernel_size=1, stride=1, padding=1)
            self.ld_point_layer = nn.Conv2d(in_channels=num_lms * 2, out_channels=num_lms*2, kernel_size=4)
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
        if self.input_size == 128:
            score_x = F.avg_pool2d(x, 4)

        elif self.input_size == 64:
            score_x = F.avg_pool2d(x, 2)

        xy = self.ld_point_layer(xy)
        xy = xy.view(-1,self.num_lms*2,1)
        score_x = score_x.view(score_x.size(0), -1)
        score = self.score_layer(score_x)
        x_pred = xy[:,::2,:]
        y_pred = xy[:,1::2,:]
        if torch.onnx.is_in_onnx_export():
            print("顺序： x_pred, y_pred, score")
            # return x1,x2,x3,score
            return x_pred, y_pred, score
        return x_pred, y_pred, score


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

class Pip_mbnetv3_small_fpn2(nn.Module):
    def __init__(self, num_lms=5, input_size=64):
        super(Pip_mbnetv3_small_fpn2, self).__init__()
        self.num_lms = num_lms
        self.input_size = input_size
        self.features = mobilenetv3_small_light()



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

class Pip_mbnetv3_small_light(nn.Module):
    def __init__(self, mbnet, num_lms=5, input_size=128):
        super(Pip_mbnetv3_small_light, self).__init__()
        self.num_lms = num_lms
        self.input_size = input_size
        self.features = mbnet

        # self.cls_layer = nn.Conv2d(576, num_lms, kernel_size=1, stride=1, padding=0)
        self.x_layer = nn.Conv2d(288, num_lms, kernel_size=1, stride=1, padding=0)
        self.y_layer = nn.Conv2d(288, num_lms, kernel_size=1, stride=1, padding=0)
        self.x_liner = nn.Linear(16,1)
        self.y_liner = nn.Linear(16,1)

        self.score_layer = nn.Linear(576,1)

        # nn.init.normal_(self.cls_layer.weight, std=0.001)
        # if self.cls_layer.bias is not None:
        #     nn.init.constant_(self.cls_layer.bias, 0)

        nn.init.normal_(self.x_layer.weight, std=0.001)
        if self.x_layer.bias is not None:
            nn.init.constant_(self.x_layer.bias, 0)

        nn.init.normal_(self.y_layer.weight, std=0.001)
        if self.y_layer.bias is not None:
            nn.init.constant_(self.y_layer.bias, 0)

        nn.init.normal_(self.score_layer.weight, std=0.001)
        if self.score_layer.bias is not None:
            nn.init.constant_(self.score_layer.bias, 0)

        nn.init.normal_(self.x_liner.weight, std=0.001)
        if self.x_liner.bias is not None:
            nn.init.constant_(self.x_liner.bias, 0)

        nn.init.normal_(self.y_liner.weight, std=0.001)
        if self.y_liner.bias is not None:
            nn.init.constant_(self.y_liner.bias, 0)

    def forward(self, x):
        x = self.features(x)
        # x1 = self.cls_layer(x)
        x2 = self.x_layer(x)
        x3 = self.y_layer(x)

        x2 = x2.view(-1,5,16)
        x3 = x3.view(-1,5,16)

        x2 = self.x_liner(x2)
        x3 = self.y_liner(x3)
        # x2 = F.avg_pool2d(x2, 4)
        # x3 = F.avg_pool2d(x3, 4)
        score_x = F.avg_pool2d(x, 4)
        score_x = score_x.view(score_x.size(0), -1)
        score = self.score_layer(score_x)

        if torch.onnx.is_in_onnx_export():
            print("顺序： x  ,y  score")
            # return x1,x2,x3,score
            return x2,x3,score
        # return x1, x2, x3, score
        return x2,x3,score

class Pip_mbnetv2_small(nn.Module):
    def __init__(self, mbnet, num_lms=5, input_size=128):
        super(Pip_mbnetv2_small, self).__init__()
        self.num_lms = num_lms
        self.input_size = input_size
        self.features = mbnet

        # self.cls_layer = nn.Conv2d(576, num_lms, kernel_size=1, stride=1, padding=0)
        self.x_layer = nn.Conv2d(240, num_lms, kernel_size=1, stride=1, padding=0)
        self.y_layer = nn.Conv2d(240, num_lms, kernel_size=1, stride=1, padding=0)
        self.x_liner = nn.Linear(16,1)
        self.y_liner = nn.Linear(16,1)

        self.score_layer = nn.Linear(240,1)

        # nn.init.normal_(self.cls_layer.weight, std=0.001)
        # if self.cls_layer.bias is not None:
        #     nn.init.constant_(self.cls_layer.bias, 0)

        nn.init.normal_(self.x_layer.weight, std=0.001)
        if self.x_layer.bias is not None:
            nn.init.constant_(self.x_layer.bias, 0)

        nn.init.normal_(self.y_layer.weight, std=0.001)
        if self.y_layer.bias is not None:
            nn.init.constant_(self.y_layer.bias, 0)

        nn.init.normal_(self.score_layer.weight, std=0.001)
        if self.score_layer.bias is not None:
            nn.init.constant_(self.score_layer.bias, 0)

        nn.init.normal_(self.x_liner.weight, std=0.001)
        if self.x_liner.bias is not None:
            nn.init.constant_(self.x_liner.bias, 0)

        nn.init.normal_(self.y_liner.weight, std=0.001)
        if self.y_liner.bias is not None:
            nn.init.constant_(self.y_liner.bias, 0)

    def forward(self, x):
        x = self.features(x)
        # x1 = self.cls_layer(x)
        x2 = self.x_layer(x)
        x3 = self.y_layer(x)

        x2 = x2.view(-1,5,16)
        x3 = x3.view(-1,5,16)

        x2 = self.x_liner(x2)
        x3 = self.y_liner(x3)
        # x2 = F.avg_pool2d(x2, 4)
        # x3 = F.avg_pool2d(x3, 4)
        score_x = F.avg_pool2d(x, 4)
        score_x = score_x.view(score_x.size(0), -1)
        score = self.score_layer(score_x)

        if torch.onnx.is_in_onnx_export():
            print("顺序： x  ,y  score")
            # return x1,x2,x3,score
            return x2,x3,score
        # return x1, x2, x3, score
        return x2,x3,score


def load_model(model, checkpoint):
    model_CKPT = torch.load(checkpoint)
    model_dict = model.state_dict()
    pretrained_dict = model_CKPT
    # 将不在model中的参数过滤掉
    new_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict.keys()}
    new_dict.pop("classifier.weight")
    new_dict.pop("classifier.bias")
    model_dict.update(new_dict)
    model.load_state_dict(model_dict)

    return model

if __name__ == '__main__':
    # net = Pip_mbnetv3_small()
    # x = torch.randn(2,3,128,128)
    # net(x)
    # dummy_input = torch.randn(1, 3, 128, 128, device='cpu')
    # torch.onnx.export(net, dummy_input, "model.onnx", export_params=True)

    from mobilenetv3 import mobilenetv3_small
    from mobilenetv2 import mobilenet_v2
    # from mobilenetv3_fpn import MobileNetV3_Small_FPN

    # mbnet = MobileNetV3_Small_FPN()
    mbnet = mobilenetv3_small()
    # mbnet = load_model(mbnet, '300_act3_mobilenetv3_small.pth')
    # net = Pip_mbnetv3_small_light(mbnet=mbnet, num_lms=5, input_size=128)
    x = torch.randn(1, 3, 64, 64)
    # net(x)
    # mbnet = mobilenet_v2(pretrained_path="../mobilenetv2_0.5-eaa6f9ad.pth", num_classes=1000, width_mult=0.5)
    # net = Pip_mbnetv3_small_fpn(mbnet, num_lms=5, input_size=64)
    net = Pip_mbnetv3_small(mbnet=mbnet, num_lms=5,input_size=64)
    # net = Pip_mbnetv3_small_ld(mbnet=mbnet, input_size=64)
    net.eval()
    net(x)
    flops, params = profile(net, (x,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))