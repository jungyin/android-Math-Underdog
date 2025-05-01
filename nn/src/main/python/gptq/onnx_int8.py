from onnxruntime.quantization import quantize_static, QuantType,QuantFormat
from onnxruntime.quantization import CalibrationDataReader
import numpy as np

class CalibDataReader:
    def __init__(self, calibration_data_paths):
        self.calibration_data_paths = calibration_data_paths
        self.data = np.load(calibration_data_paths,allow_pickle=True)
        kk = self.data[0][0]
        self.index = 0

    def get_next(self):
        if self.index < len(self.data):
            data = self.data[0][self.index]
            self.index += 1
            return data  # 根据你的模型调整 'input_name'
        else:
            return None

    def rewind(self):
        self.index = 0

# 初始化校准数据读取器
calibration_data_reader = CalibDataReader("cache.npy")

# 输入输出路径
input_model_path = './model_file/qwen2-code-0.5b/onnx/model32.onnx'
output_model_path = './model_file/qwen2-code-0.5b/onn_qint8/model.onnx'



# 执行量化
quantize_static(
    input_model_path,
    output_model_path,
    calibration_data_reader,
    quant_format=QuantFormat.QDQ,  # 或者使用 QuantFormat.ONNX 依据需要
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    per_channel=True,  # 可选：逐通道量化权重
    use_external_data_format = True,
    # optimize_model=False,  # 关闭模型优化，如果不需要
    # enable_subgraph=False  # 如果不需要子图支持，可以关闭
)