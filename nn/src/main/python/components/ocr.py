from tokenizers import Tokenizer
from transformers import AutoTokenizer
from PIL import Image
import requests
import cv2
import numpy as np
import torch
import sys

from infer.latex_ocr.openvino_infer import LatexMoelRun
  
# from infer.latex_ocr.onnx_infer import LatexMoelRun

p = "./assets/latex_ocr/"
img = "test.png"


model = LatexMoelRun(p)

frame = cv2.imread(img)
out = model.greedy_search(frame)


# tokenizer = tokenizer = Tokenizer.from_file(p+str("tokenizer_config.json"))
tokenizer = tokenizer = AutoTokenizer.from_pretrained(p, max_len=296)

moutstr = tokenizer.decode(out)
print("原始:",moutstr)
print("结束",moutstr.replace('\\[','\\begin{align*}').replace('\\]','\\end{align*}'))

print(1)
