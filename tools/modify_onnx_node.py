import onnx


model = onnx.load("../onnx/pipnet_mbv1_blurness_v5_64_retinaface_50ep.onnx")

model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = 1
model.graph.output[1].type.tensor_type.shape.dim[0].dim_value = 1

model.graph.output[2].type.tensor_type.shape.dim[0].dim_value = 1
model.graph.output[2].type.tensor_type.shape.dim[2].dim_value = 1
model.graph.output[2].type.tensor_type.shape.dim[3].dim_value = 1



# print(model.graph.output[1].name)

onnx.save(model, "pipnet_mbv1_blurness_v5_64_retinaface_50ep_new.onnx")