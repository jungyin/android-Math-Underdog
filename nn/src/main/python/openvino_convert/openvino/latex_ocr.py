
import openvino as ov
import onnxruntime  as ort
base_path = "./model_file/latex-ocr/"
input_model = base_path+"onnx/ocr_encoder.onnx"
input_model = base_path+"onnx/ocr_decoder.onnx"

output_model = base_path+"openvino/ocr_encoder.xml"
output_model = base_path+"openvino/ocr_decoder.xml"

# 加载ONNX模型
session = ort.InferenceSession(input_model)

# 获取模型的输入和输出名称及其形状
print("Model Inputs:")
for input_name in session.get_inputs():
    print(f"Name: {input_name.name}, Shape: {input_name.shape}, Type: {input_name.type}")

print("\nModel Outputs:")
for output_name in session.get_outputs():
    print(f"Name: {output_name.name}, Shape: {output_name.shape}, Type: {output_name.type}")
ov_model = ov.convert_model(input_model)

###### Option 1: Save to OpenVINO IR:



# save model to OpenVINO IR for later use
ov.save_model(ov_model, output_model)

###### Option 2: Compile and infer with OpenVINO:

# compile model
# compiled_model = ov.compile_model(ov_model)

# # prepare input_data
# import numpy as np
# input_data = np.random.rand(1, 3, 224, 224)

# # run inference
# result = compiled_model(input_data)