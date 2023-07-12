import onnx
import torch
from lib.networks import Pip_mbnetv3_small,Pip_mbnetv2_small,Pip_mbnetv3_small_ld,Pip_mbnetv3_small_fpn
from lib.mobilenetv3 import mobilenetv3_small_light, mobilenetv3_small
from lib.mobilenetv3_fpn import MobileNetV3_Small_FPN
from lib.retinaface import RetinaFace,RetinaFace_Blurness
from lib.mobilenetv2 import mobilenet_v2

if __name__ == '__main__':
    model_path = "../snapshots/Widerface/pipnet_mbv1_blurness_v5_64_retinaface/epoch70.pth"
    onnx_name = "../onnx/pipnet_mbv1_blurness_v5_64_retinaface_70ep.onnx"


    # mbnet = MobileNetV3_Small_FPN()
    # model = Pip_mbnetv3_small_fpn(mbnet=mbnet, input_size=64)
    cfg_mnet = {
        'name': 'mobilenet0.25',
        'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
        'in_channel': 32,
        'out_channel': 64
    }
    model = RetinaFace_Blurness(cfg=cfg_mnet)
    # model = Pip_mbnetv2_small(mbnet=mbnet)
    model_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(model_dict)
    model.eval()

    input_names = ['input']
    output_names = ['x_pred', 'y_pred', 'score']
    tensor_data = torch.randn((1, 3, 64, 64))
    torch.onnx.export(
        model,
        tensor_data,
        onnx_name,
        keep_initializers_as_inputs=False,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None,
        opset_version=11)