
import nncf.parameters
from openvino.runtime import serialize
import torch
import nncf
import openvino as ov
import torch
import pickle
from torch .utils.data import Dataset
import numpy as np

openvino = "./model_file/qwen2-3b/openvino/model.xml"
openvino_int8 = "./model_file/qwen2-3b/openvino_qint8/model.xml"

# 预处理 qwen2的输入数据
def mypre(
   input_ids,attention_mask=None
):

  # create position_ids on the fly for batch generation
    attention_mask = torch.from_numpy(attention_mask)
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    

    model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids.numpy(),
            "attention_mask": attention_mask.numpy(),
        }
    )
    return model_inputs

class MyDataset(Dataset):
    def __init__(self):
        super().__init__()
        data = np.load("cache.npy",allow_pickle=True)
        self.input_data = data
    def __len__(self):
        # return int( len(self.items) / 3)
        return len(self.input_data)

    def __getitem__(self, idx):
        input_data=  self.input_data[idx]
        # input_ids = input_data["input_ids"]
        input_ids = input_data["input_ids"]
        p_a = mypre(input_ids,np.zeros_like(input_ids))

        return input_ids[0],p_a["attention_mask"][0],p_a["position_ids"][0],input_data["past_key_values"]


def transform_fn(data_item):
    images, label = data_item

    position_ids = torch.unsqueeze(torch.arange(0,images.shape[2]),0)
    return images[0],torch.zeros_like(images[0]),position_ids

calibration_loader = torch.utils.data.DataLoader(MyDataset())
calibration_dataset = nncf.Dataset(calibration_loader)

core = ov.Core()
model = core.read_model(model=openvino)

quantized_model = nncf.quantize(
	    model, 
	    calibration_dataset,
        # mode=nncf.QuantizationMode.FP8_E4M3,
	    preset=nncf.QuantizationPreset.PERFORMANCE,
        # target_device = nncf.TargetDevice.CPU,
        model_type=nncf.ModelType.TRANSFORMER
	    )

serialize(quantized_model, openvino_int8)
