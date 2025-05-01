import openvino as ov
import numpy as np
from openvino.runtime import Core
import time
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor

onnx = "./yolov11/onnx/yolo11m.onnx"

openvino = "./yolov11/openvino/model32.xml"

openvino_int8 = "./yolov11/openvino_int8/model32.xml"

# ov_model = ov.convert_model(onnx,input=[("images",[1,3,640,640])])

# ov.save_model(ov_model, openvino,compress_to_fp16=False)



core = Core()
devices = core.available_devices
device = 'CPU'
if 'NPU' in devices:
    device = 'NPU'
elif 'GPU':
    device = 'GPU'

model = core.read_model(model=openvino)


compiled_model = core.compile_model(model, device)
time123 = 0
for i in range(0,150):

    # gpu及cpu允许动态输入长度
    # chunk = 47 + i

    input_ids = ov.Tensor(array = np.random.rand(1, 3,640,640).astype(np.float32),shared_memory=True)

    infer_request = compiled_model.create_infer_request()

    infer_request.set_tensor("images",input_ids)
    infer_request.infer()
    # infer_request.wait()
    # Get output tensor for model with one output
    output = infer_request.get_output_tensor()
    output_buffer = output.data

    print("end1",(time.time()-time123))
    time123=time.time()