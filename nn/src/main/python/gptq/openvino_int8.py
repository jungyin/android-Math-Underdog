
import nncf.parameters
from openvino.runtime import serialize
import torch
import nncf
import openvino as ov
import torch
import pickle
from torch .utils.data import Dataset


openvino = "./qwen2-code-0.5b/openvino/model32.xml"
openvino_int8 = "./qwen2-code-0.5b/openvino_qint8/model32.xml"

class MyDataset(Dataset):
    def __init__(self):
        super().__init__()

        with open('cache.pkl', 'rb') as f:
            self.items,self.labels = pickle.load(f)
        i = 1
    def __len__(self):
        # return int( len(self.items) / 3)
        return 10

    def __getitem__(self, idx):
        item = self.items[idx]
        return item,self.labels[idx]


def transform_fn(data_item):
    images, label = data_item

    position_ids = torch.unsqueeze(torch.arange(0,images.shape[2]),0)
    return images[0],torch.zeros_like(images[0]),position_ids

calibration_loader = torch.utils.data.DataLoader(MyDataset())
calibration_dataset = nncf.Dataset(calibration_loader, transform_fn)

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
