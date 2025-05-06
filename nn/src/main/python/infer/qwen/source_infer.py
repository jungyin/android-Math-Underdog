
import numpy as np
import onnxruntime as ort
from onnxruntime.capi import _pybind_state as C
from .base_infer import BaseMoelRun
import time
from transformers import AutoModelForCausalLM
import torch
class QwenMoelRun(BaseMoelRun):
    def __init__(self,model_assets):
        super().__init__(model_assets)
        # 模型所在路径
        model_f = "D:\\code\\transformer_models\\"
        model = "models--Qwen--Qwen2.5-3B-Instruct"
        self.model_path = model_f + model


        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto",
            local_files_only=True
        )
        model.base_model.kk = True
        model.kk = True
        self.model = model
        self.device = model.device



    # 这里模拟ForCausalLM方法
    def runForCausalLM(self ,input_ids,past_key_values=None):
       
        inputs = self.prepare_inputs_for_generation(input_ids,None,None,past_key_values,None)
        # inputs = {"input_ids": inputs['input_ids'],"attention_mask":inputs["attention_mask"],"position_ids":inputs["position_ids"],"past_key_values.1":inputs["past_key_values"]}
        with torch.no_grad():
            output = self.model(torch.from_numpy(inputs["input_ids"]).to(self.device),torch.from_numpy(inputs["attention_mask"]).to(self.device),torch.from_numpy(inputs["position_ids"]).to(self.device),past_key_values)
 
            return output[0].cpu().detach().numpy(),output[1]

if __name__ == "__main__":
    qwen = QwenMoelRun()
    