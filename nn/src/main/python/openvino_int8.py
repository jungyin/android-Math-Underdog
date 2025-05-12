
import nncf.parameters
from openvino.runtime import serialize
import torch
import nncf
import openvino as ov
import torch
import pickle
from torch .utils.data import Dataset
import numpy as np

from infer.qwen_0_5b.base_infer import BaseMoelRun

openvino = "./model_file/qwen2-code-0.5b/openvino/model32.xml"
openvino_int8 = "./model_file/qwen2-code-0.5b/openvino_qint8/model32.xml"


class MyDataset(Dataset):
    def __init__(self):
        super().__init__()
        data = np.load("./cache.npy",allow_pickle=True)
        
        self.input_data = data[0]
        self.output_data = data[1]
    
    def __len__(self):
        # return int( len(self.items) / 3)
        return len(self.input_data) 

    def __getitem__(self, idx):
        item = self.input_data[idx]
        input = BaseMoelRun.prepare_inputs_for_generation(None,item["input_ids"],None,None,item["past_key_values"])

        outputs = self.output_data[idx]
        return  input["input_ids"][0],input["attention_mask"][0],input["position_ids"][0],input["past_key_values"]


def transform_fn(data_item):
    images, label = data_item
    return images["input_ids"][0],images["attention_mask"][0],images["position_ids"][0],images["past_key_values"]

myDataset = MyDataset()
ss = myDataset.__len__()
kk = myDataset.__getitem__(0)
calibration_loader = torch.utils.data.DataLoader(myDataset)
calibration_dataset = nncf.Dataset(calibration_loader)

core = ov.Core()
model = core.read_model(model=openvino)

quantized_model = nncf.quantize(
	    model, 
	    calibration_dataset,
        # mode=nncf.QuantizationMode.FP8_E4M3,
	    preset=nncf.QuantizationPreset.PERFORMANCE,
        # target_device = nncf.TargetDevice.CPU,
        model_type=nncf.ModelType.TRANSFORMER,
        
	    )

serialize(quantized_model, openvino_int8)
