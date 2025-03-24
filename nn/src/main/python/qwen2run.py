
from np_logits.logits_process import RepetitionPenaltyLogitsProcessor,TemperatureLogitsWarper,TopKLogitsWarper,TopPLogitsWarper , MaxLengthCriteria ,EosTokenCriteria
import numpy as np
import onnxruntime as ort
from nputils import softmax,multinomial_numpy


class QwenMoelRun():
    def __init__(self):
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

        input_names = [input.name for input in self.model.get_inputs()]
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
    