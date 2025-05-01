from tokenizers import Tokenizer
from transformers import AutoTokenizer
from PIL import Image
import requests
import cv2
import numpy as np
import torch
from infer.latex_ocr.onnx_infer import LatexMoelRun

p = "D:\\code\\android\\android-Math-Underdog\\nn\\src\\main\\python\\assets\\latex_ocr/"
img = "test.png"


model = LatexMoelRun(p)

frame = cv2.imread(img)
out = model.greedy_search(frame)


# tokenizer = tokenizer = Tokenizer.from_file(p+str("tokenizer_config.json"))
tokenizer = tokenizer = AutoTokenizer.from_pretrained(p, max_len=296)

moutstr = tokenizer.decode(out)
print(moutstr.replace('\\[','\\begin{align*}').replace('\\]','\\end{align*}'))

print(1)
