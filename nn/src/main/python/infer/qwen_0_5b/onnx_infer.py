
import numpy as np
import onnxruntime as ort

from .base_infer import BaseMoelRun

class QwenMoelRun(BaseMoelRun):
    def __init__(self):
        super().__init__()
        # 模型参数的加载
        self.pad_token_id = 151643
        self.bos_token_id = 151643
        self.eos_token_id = [151645,151643]
        self.max_position_embeddings=32768
        self.max_len = 0
        # 是否存在最大长度
        self.has_default_max_length = True
        # 是否存在最短长度
        self.has_default_min_length = True
        # 最长tokean数
        self.max_new_tokens=512
        # 由于没有限制最小长度，这里配为0
        self. min_length = 0
         # 模型所在路径
        model = "qwen2-code-0.5b"
        model_type="onnx"
        self.model_path ="./" + model + "/" + model_type + "/" + "model.onnx"
        self.model_path ="./" + model + "/" + model_type + "/" + "model32.onnx"
        self.lm_model_path ="./" + model + "/" + model_type + "/" + "lm_model32.onnx"
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        # session_options.enable_cuda_graph = True  # 如果需要启用 CUDA 图形优化
        # session_options.gpu_id = 0  # 指定使用第 0 块 GPU
        # 有一个linear层无法导出来，这里需要手动额外加载使用
        self.lm_head = ort.InferenceSession(self.lm_model_path, sess_options=session_options, providers=['CUDAExecutionProvider'])
        # 模型本体的加载
        print("Model is valid and supported by the current ONNX Runtime.")
        self.model = ort.InferenceSession(self.model_path, sess_options=session_options, providers=['CUDAExecutionProvider'])


    # 这里模拟ForCausalLM方法
    def runForCausalLM(self ,input_ids):

        inputs = self.prepare_inputs_for_generation(input_ids,None,None,None)
        # inputs = {"input_ids": inputs['input_ids'],"attention_mask":inputs["attention_mask"],"position_ids":inputs["position_ids"]}
        output_names = [output.name for output in self.model.get_outputs()]
        outputs=self.model.run(output_names,inputs)
        hidden_states = outputs[0]

        llm_inputs = {"input_0": hidden_states}
        llm_output_names = [output.name for output in self.lm_head.get_outputs()]
        logits = self.lm_head.run(llm_output_names,llm_inputs)
        logits = logits[0].astype(np.float32)
        return logits

if __name__ == "__main__":
    qwen = QwenMoelRun()
    