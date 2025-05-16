import numpy as np
from openvino.runtime import Core
from .base_infer import BaseMoelRun
import openvino.runtime as ov

class QwenMoelRun(BaseMoelRun):
    def __init__(self,model_assets):
        super().__init__(model_assets)

    
         # 模型所在路径
        model = "qwen2-0.5b"
        # model = "qwen2-3b"
        model_type="openvino_test"
        # model_type="openvino_qint8"
        model_path = self.model_f  + model + "/" + model_type + "/" + "model.xml"
        maxinputsize = 256
        self.maxinputsize = maxinputsize
   
        core = Core()

        self.core = core

        devices = core.available_devices
        device = 'CPU'
        if 'NPU' in devices :
            device = 'NPU'
        elif 'GPU'in devices:
            device = 'GPU'
        self.device = device

        model = core.read_model(model=model_path)

        input_shapes = {
            "input_ids": ov.PartialShape([1, maxinputsize]),
            "attention_mask":ov.PartialShape([1, maxinputsize]),
            "position_ids":ov.PartialShape([1, maxinputsize]),
            "position_ids":ov.PartialShape([1, maxinputsize]),
            "past_key_values.1":ov.PartialShape([24,2,1,2,maxinputsize,64]),
            "clip_index":ov.PartialShape([1,4]),

        }

        model.reshape(input_shapes)

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
        # if(not (past_key_values is None)):
        #     past_key_values = past_key_values[:,:,:,:,:-1,:]


        inputs = self.prepare_inputs_for_generation(input_ids,None,None,past_key_values,None)

        # if(len( self.cacheinput)<500):
        #     self.cacheinput .append ({"input_ids":inputs['input_ids'],"past_key_values":inputs["past_key_values"]})
        # else:
        #     self.stopGenerate()
       
       
        shared_memory = True
        input_ids = ov.Tensor(array = inputs["input_ids"],shared_memory=shared_memory)
        attention_mask = ov.Tensor(array = inputs["attention_mask"],shared_memory=shared_memory)
        position_ids = ov.Tensor(array = inputs["position_ids"],shared_memory=shared_memory)
        past_key_values = ov.Tensor(array = inputs["past_key_values"],shared_memory=shared_memory)
        # past_key_values = ov.Tensor(array = np.ones([24,2,1,2,46,64],dtype=np.float32),shared_memory=True)
        
        infer_request = self.model.create_infer_request()
        infer_request.set_tensor("input_ids",input_ids)
        infer_request.set_tensor("attention_mask",attention_mask)
        infer_request.set_tensor("position_ids",position_ids)
        # if needpk:
        infer_request.set_tensor("past_key_values.1",past_key_values)
        infer_request.infer(share_inputs= shared_memory,share_outputs=shared_memory)

        output_tensor_names = self.model.outputs

        output = infer_request.get_tensor(output_tensor_names[0]).data
        past_buffer = infer_request.get_tensor(output_tensor_names[1]).data
        
        del input_ids
        del position_ids
        del attention_mask
        del past_key_values
        del output_tensor_names
        del infer_request

        return output,past_buffer

if __name__ == "__main__":
    qwen = QwenMoelRun()
    