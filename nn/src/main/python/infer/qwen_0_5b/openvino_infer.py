import numpy as np
from openvino.runtime import Core
from .base_infer import BaseMoelRun
import openvino as ov

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
        model_type="openvino"
        model_path ="./" + model + "/" + model_type + "/" + "model32.xml"
        
        model_type="openvino"
        lm_model_path ="./" + model + "/" + model_type + "/" + "lm_model32.xml"

        core = Core()

        self.core = core

        devices = core.available_devices
        device = 'CPU'
        if 'NPU' in devices and False:
            device = 'NPU'
        elif 'GPU'in devices:
            device = 'GPU'
        self.device = device

        model = core.read_model(model=model_path)
        llm_model = core.read_model(model=lm_model_path)

        self.model = core.compile_model(model, device)
        self.lm_head = core.compile_model(llm_model, device)


        self.cacheinput = []
        self.cacheoutput = []
  
    def reshape_inputids(self,model,llm,shape = [1,-1]):
        input_shapes = {
            "input_ids": ov.PartialShape(shape),
            "attention_mask": ov.PartialShape(shape),
            "position_ids":ov. PartialShape(shape)
        }
        model.reshape(input_shapes)
        llm.reshape([1,shape[1],896])

    # 这里模拟ForCausalLM方法
    def runForCausalLM(self ,input_ids):
        inputs = self.prepare_inputs_for_generation(input_ids,None,None,None)

        chunk = input_ids[-1]

        
        self.cacheinput .append (inputs['input_ids'])

        input_ids = ov.Tensor(array = inputs["input_ids"],shared_memory=True)
        attention_mask = ov.Tensor(array = inputs["attention_mask"],shared_memory=True)
        position_ids = ov.Tensor(array = inputs["position_ids"],shared_memory=True)

        infer_request = self.model.create_infer_request()
        infer_request.set_tensor("input_ids",input_ids)
        infer_request.set_tensor("attention_mask",attention_mask)
        infer_request.set_tensor("position_ids",position_ids)
        infer_request.infer()
        output = infer_request.get_output_tensor()
        output_buffer = ov.Tensor(array = output.data,shared_memory=True)

        self.cacheoutput.append(output.data)

        infer_lm_request = self.lm_head.create_infer_request()
        infer_lm_request.set_tensor("input_0",output_buffer)
        infer_lm_request.infer()
        output = infer_lm_request.get_output_tensor()
        logits = output.data
        logits = logits
        return logits

if __name__ == "__main__":
    qwen = QwenMoelRun()
    