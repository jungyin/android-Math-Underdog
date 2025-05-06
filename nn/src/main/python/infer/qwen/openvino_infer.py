import numpy as np
from openvino.runtime import Core
from .base_infer import BaseMoelRun
import openvino as ov

class QwenMoelRun(BaseMoelRun):
    def __init__(self,model_assets):
        super().__init__(model_assets)

    
         # 模型所在路径
        model = "qwen2-code-0.5b"
        model = "qwen2-3b"
        model_type="openvino"
        # model_type="openvino_qint8"
        model_path = self.model_f  + model + "/" + model_type + "/" + "model.xml"
        
   
        core = Core()

        self.core = core

        devices = core.available_devices
        device = 'CPU'
        # if 'NPU' in devices and False:
        #     device = 'NPU'
        # elif 'GPU'in devices:
        #     device = 'GPU'
        self.device = device
        self.device = "GPU"

        model = core.read_model(model=model_path)

        self.model = core.compile_model(model, device)


        self.cacheinput = []
        self.cacheoutput = []

        # 获取所有输入的信息
        input_names = [input.any_name for input in self.model.inputs]

        # 打印所有输入的名字
        for name in input_names:
            print(f"Input name: {name}")
  
    def reshape_inputids(self,model,llm,shape = [1,-1]):
        input_shapes = {
            "input_ids": ov.PartialShape(shape),
            "attention_mask": ov.PartialShape(shape),
            "position_ids":ov. PartialShape(shape)
        }
        model.reshape(input_shapes)
        llm.reshape([1,shape[1],896])

    # 这里模拟ForCausalLM方法
    def runForCausalLM(self ,input_ids,past_key_values=None):
        inputs = self.prepare_inputs_for_generation(input_ids,None,None,past_key_values,None)

        chunk = input_ids[-1]

        
        needpk = inputs["past_key_values"].shape[-2]!=0
        if(len( self.cacheinput)<500):
            self.cacheinput .append ({"input_ids":inputs['input_ids'],"past_key_values":inputs["past_key_values"]})
        else:
            self.stopGenerate()

        input_ids = ov.Tensor(array = inputs["input_ids"],shared_memory=True)
        attention_mask = ov.Tensor(array = inputs["attention_mask"],shared_memory=True)
        position_ids = ov.Tensor(array = inputs["position_ids"],shared_memory=True)
        past_key_values = ov.Tensor(array = inputs["past_key_values"],shared_memory=True)
        # past_key_values = ov.Tensor(array = np.ones([24,2,1,2,46,64],dtype=np.float32),shared_memory=True)
        
        infer_request = self.model.create_infer_request()
        infer_request.set_tensor("input_ids",input_ids)
        infer_request.set_tensor("attention_mask",attention_mask)
        infer_request.set_tensor("position_ids",position_ids)
        # if needpk:
        infer_request.set_tensor("past_key_values.1",past_key_values)
        infer_request.infer()

        output_tensor_names = self.model.outputs

        output = infer_request.get_tensor(output_tensor_names[0])
        past_key_values = infer_request.get_tensor(output_tensor_names[1])


        output_buffer = ov.Tensor(array = output.data,shared_memory=True).data
        past_buffer = ov.Tensor(array = past_key_values.data,shared_memory=True).data

        return output_buffer,past_buffer

if __name__ == "__main__":
    qwen = QwenMoelRun()
    