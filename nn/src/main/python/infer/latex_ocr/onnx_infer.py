
import numpy as np
import onnxruntime as ort

from .base_infer import BaseMoelRun

class LatexMoelRun(BaseMoelRun):
    def __init__(self,model_assets):
        super().__init__(model_assets)
         # 模型所在路径
        model = "latex-ocr"
        model_type="onnx"

        encoder_path = self.model_f + model + "/" + model_type + "/" + "ocr_encoder.onnx"
        decoder_path = self.model_f + model + "/" + model_type + "/" + "ocr_decoder.onnx"
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        # session_options.enable_cuda_graph = True  # 如果需要启用 CUDA 图形优化
        # session_options.gpu_id = 0  # 指定使用第 0 块 GPU
        self.encode = ort.InferenceSession(encoder_path, sess_options=session_options, providers=['CUDAExecutionProvider'])
        self.decode = ort.InferenceSession(decoder_path, sess_options=session_options, providers=['CUDAExecutionProvider'])


    def encoder(self,pixel_values):
        inputs = {"pixel_values": pixel_values}
        output_names = [output.name for output in self.encode.get_outputs()]
        outputs=self.encode.run(output_names,inputs)
        return outputs[0]
    def decoder(self,input_ids,attention_mask,encoder_hidden_states):
       
        input_names = [output.name for output in self.decode.get_inputs()]
        inputs = {"input_ids": input_ids,"attention_mask":attention_mask,"encoder_hidden_states":encoder_hidden_states}
        output_names = [output.name for output in self.decode.get_outputs()]
        outputs=self.decode.run(output_names,inputs)
        return outputs[0]
        

if __name__ == "__main__":
    latex = LatexMoelRun()
    