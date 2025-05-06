
import numpy as np
import onnxruntime as ort
from jinja2.exceptions import TemplateError
from onnxruntime.capi import _pybind_state as C
from .base_infer import BaseMoelRun
import time
class QwenMoelRun(BaseMoelRun):
    def __init__(self,model_assets):
        super().__init__(model_assets)
         # 模型所在路径
        model = "qwen2-code-0.5b"
        model_type="onnx"
        # model_type="onnx_qint8"
        self.model_path = self.model_f + model + "/" + model_type + "/" + "model32.onnx"
        # self.model_path = "D:\code\py\qwen2demo_py\onnx\math-1.5b\model.onnx"
        self.model_path = "D:/code\py\qwen2demo_py\onnx\qwen2_3b\model.onnx"

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        # session_options.enable_cuda_graph = True  # 如果需要启用 CUDA 图形优化
        # 模型本体的加载
        print("Model is valid and supported by the current ONNX Runtime.")

        # session_options =  C.get_default_session_options()
        # sess = C.InferenceSession(session_options, self.model_path, True, False)
    
        self.model = ort.InferenceSession(self.model_path, sess_options=session_options, providers=['CUDAExecutionProvider'])


    # 这里模拟ForCausalLM方法
    def runForCausalLM(self ,input_ids,past_key_values=None):
       
        inputs = self.prepare_inputs_for_generation(input_ids,None,None,past_key_values,None)
        inputs = {"input_ids": inputs['input_ids'],"attention_mask":inputs["attention_mask"],"position_ids":inputs["position_ids"],"past_key_values.1":inputs["past_key_values"]}
        output_names = [output.name for output in self.model.get_outputs()]
        outputs=self.model.run(output_names,inputs)
      
        return outputs

if __name__ == "__main__":
    qwen = QwenMoelRun()
    