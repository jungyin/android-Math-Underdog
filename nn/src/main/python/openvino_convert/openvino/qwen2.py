import openvino as ov
import numpy as np
from openvino.runtime import Core
import time
from torch.utils.data import Dataset

onnx = "./qwen2-code-0.5b/onnx/model.onnx"
onnx = "D:/code/py/qwen2demo_py/onnx/qwen2_0.5b_test/model.onnx"
openvino = "./qwen2-code-0.5b/openvino_test/model.xml"

# ov_model = ov.convert_model(onnx,input=[("input_ids",[1,-1]),("attention_mask",[1,-1]),("position_ids",[1,-1])])

# ov.save_model(ov_model, openvino,compress_to_fp16=False)

# ov_model = ov.convert_model(onnxllm,input=[("input_0",[1,-1,896])])

# ov.save_model(ov_model, openvinollm,compress_to_fp16=False)




core = Core()
devices = core.available_devices
device = 'CPU'
if 'NPU' in devices:
    device = 'NPU'
elif 'GPU':
    device = 'GPU'
device = 'CPU'
 
# npu不允许动态输入长度
chunk = -1
# 如果要用npu，就上个512固定
chunk = 1024

model = core.read_model(model=openvino)


input_shapes = {
    "input_ids": ov.PartialShape([1, chunk]),
    "attention_mask": ov.PartialShape([1, chunk]),
    "position_ids":ov. PartialShape([1, chunk])
}

model.reshape(input_shapes)

input_shapes = {
    "input_0": ov.PartialShape([1, chunk,896]),
}


compiled_model = core.compile_model(model, device)
time123 = 0
for i in range(0,150):

    # gpu及cpu允许动态输入长度
    # chunk = 47 + i

    input_ids = ov.Tensor(array = np.random.rand(1, chunk).astype(np.int64),shared_memory=True)
    attention_mask = ov.Tensor(array = np.ones([1,chunk],dtype=np.int64),shared_memory=True)
    position_ids = ov.Tensor(array = np.expand_dims(np.arange(chunk,dtype=np.int64),0),shared_memory=True)


    infer_request = compiled_model.create_infer_request()

    infer_request.set_tensor("input_ids",input_ids)
    infer_request.set_tensor("attention_mask",attention_mask)
    infer_request.set_tensor("position_ids",position_ids)
    infer_request.infer()
    # infer_request.wait()
    # Get output tensor for model with one output
    output = infer_request.get_output_tensor()
    output_buffer = output.data

    infer_lm_request = compiled_llm_model.create_infer_request()

    input_0 = ov.Tensor(array = output_buffer,shared_memory=True)
    infer_lm_request.set_tensor("input_0",input_0)
    infer_lm_request.infer()
    output = infer_lm_request.get_output_tensor()
    output_buffer = output.data

    print("end1",1/(time.time()-time123))
    time123=time.time()