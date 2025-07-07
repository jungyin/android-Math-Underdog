import numpy as np
from openvino.runtime import Core
from .base_infer import BaseMoelRun
import openvino.runtime as ov
import psutil
import os 
class LatexMoelRun(BaseMoelRun):
    def __init__(self,model_assets):
        super().__init__(model_assets)

    
         # 模型所在路径
        # model = "qwen2-code-0.5b"
        model = "latex-ocr"
        model_type="openvino"
        # model_type="openvino_qint8"
        model_de = self.model_f  + model + "/" + model_type + "/" + "ocr_decoder.xml"
        model_en = self.model_f  + model + "/" + model_type + "/" + "ocr_encoder.xml"
        
   
        core = Core()

        self.core = core

        devices = core.available_devices
        device = 'CPU'
        if 'NPU' in devices:
            device = 'GPU'
        elif 'GPU'in devices:
            device = 'GPU'
        self.device = device
        self.device = "GPU"

        model_en = core.read_model(model=model_en)
        self.model_en = core.compile_model(model_en, device)

        if(device == 'NPU'):
            device = 'GPU'

        model_de = core.read_model(model=model_de)
        self.model_de = core.compile_model(model_de, device)


    def reshape_inputids(self,model,llm,shape = [1,-1]):
        input_shapes = {
            "input_ids": ov.PartialShape(shape),
            "attention_mask": ov.PartialShape(shape),
            "position_ids":ov. PartialShape(shape)
        }
        model.reshape(input_shapes)
        llm.reshape([1,shape[1],896])

    def encoder(self,pixel_values):
        shared_memory = False
        pi = ov.Tensor(array = pixel_values,shared_memory=shared_memory)

        infer_request = self.model_en.create_infer_request()
        infer_request.set_tensor("pixel_values",pi)

        infer_request.infer(share_inputs= shared_memory,share_outputs=shared_memory)
        return infer_request.get_tensor(self.model_en.outputs[0]).data

    def decoder(self,input_ids,attention_mask,encoder_hidden_states):


        shared_memory = True
        ii = ov.Tensor(array = input_ids,shared_memory=shared_memory)
        am = ov.Tensor(array = attention_mask,shared_memory=shared_memory)
        ehs = ov.Tensor(array = encoder_hidden_states,shared_memory=shared_memory)

        infer_request = self.model_de.create_infer_request()
        infer_request.set_tensor("input_ids",ii)
        infer_request.set_tensor("attention_mask",am)
        infer_request.set_tensor("encoder_hidden_states",ehs)

        infer_request.infer(share_inputs= shared_memory,share_outputs=shared_memory)

        return infer_request.get_tensor(self.model_de.outputs[0]).data
       
        
if __name__ == "__main__":
    latex = LatexMoelRun()
    