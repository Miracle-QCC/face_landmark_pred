from lib.mobilenetv3 import mobilenetv3_small,mobilenetv3_small_light
from lib.networks import MobileNetV3_Small
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import torch




if __name__ == '__main__':
    net1 = mobilenetv3_small_light()
    net2 = MobileNetV3_Small()
    net1.eval()
    net2.eval()
    x = torch.rand((1,3,128,128))

    flops = FlopCountAnalysis(net1, x)
    print("FLOPs: ", flops.total())

    flops = FlopCountAnalysis(net2, x)
    print("FLOPs: ", flops.total())

    # 分析parameters
    # print(parameter_count_table(model))