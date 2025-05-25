from tokenizers import Tokenizer
import numpy as np

import time

# from testrun import QwenMoelRun
# from qwen2run import QwenMoelRun
# from infer.qwen.source_infer import QwenMoelRun
from infer.qwen.openvino_infer import QwenMoelRun
from infer.latex_ocr.openvino_infer import LatexMoelRun
# from infer.qwen.openvino_npu_infer import QwenMoelRun
# from infer.qwen.onnx_infer import QwenMoelRun
from transformers import AutoTokenizer
# 读取 tokenizer_config.json 文件
import json
from components.screenshot import start_screenshot_ocr
# p = "./assets/latex_ocr/"

# tokenizer = AutoTokenizer.from_pretrained(p, max_len=296)
# ocr_model = LatexMoelRun(p)

# frame = start_screenshot_ocr(ocr_model,tokenizer)


# ocrstr_ids = ocr_model.greedy_search(frame)
# ocrstr = tokenizer.decode(ocrstr_ids)

outstrs = ""

t1 = 0
times = []
dstr = []
dids = []
def print_progress(ntoken,tokenizer,lstr =""):
  """
  调用本函数将打印一个基于当前迭代次数的进度条。
  """
  global t1
  global dstr
  global dids
  if(t1!=0):
    t2 = time.time() - t1
    times.append(t2)
  t1 =time.time()
  dids.append(ntoken[0][0])
  di = np.array(dids)
  d_text = tokenizer.decode(di,skip_special_tokens=True)
  dstr.append(d_text)
  if("�" in d_text):
    d_text = ""
  else:
    dids=[]
     

  print(d_text,end='')  # 打印字符但不换行，并立即刷新输出缓冲区
  return d_text

model_path = "./assets/qewn2/"
# model_path = "D:\code\py\qwen\source\qwen2.5_1.5b_math/"
model_path = "D:\code\\transformer_models\models--Qwen--Qwen2.5-3B-Instruct/"


# 配置项
tokenizer_json = model_path+'tokenizer.json'
add_generation_prompt = True
# 测试数据
prompt= "请帮我写一个傅里叶变化公式,并使用python代码简单复现一下"
# prompt= "旧金山天气如何？"
# prompt= "今天气天何如"
# prompt= "请帮我使用vue写一个helloworld"
# prompt= "请帮我写一个傅里叶变化公式"
# prompt = '请问a+b=3,a+a=2,a,b分别是多少'
# prompt = "已知三角形 $ABC$ 的内切圆分别切边 $BC$, $CA$, $AB$ 于点 $D$, $E$, $F$。若 $AD$、$BE$、$CF$ 相交于一点 $I$（内心），求证：$$\frac{AI}{ID} + \frac{BI}{IE} + \frac{CI}{IF} = 2.$$"

messages = [
    # {"role": "system", "content": "你是一个可爱的猫娘，你的回答将在结尾添加一个喵来结束"},
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt},
    # {"role": "assistant", "content": "好的，我将为您提供如下结果的喵"},
    # {"role": "user", "content": prompt}
]



rendered=[]

conversations = [messages]



# 前处理一下输入内容
tokenizer = Tokenizer.from_file(tokenizer_json)

# out_code = np.load("out_code.npy")

out_str = []

# for code in out_code:
# out_str.append(tokenizer.decode([out_code[8][0]],skip_special_tokens=True))


model = QwenMoelRun(model_path)
for chat in conversations:
  if hasattr(chat, "messages"):
      # Indicates it's a Conversation object
      chat = chat.messages
  rendered_chat = model.compiled_template.render(
      messages=chat, add_generation_prompt=add_generation_prompt, **model.template_kwargs
  )
  rendered.append(rendered_chat)



# 测试编码和解码
encoding = tokenizer.encode_batch(
            rendered,
            add_special_tokens=True,
            is_pretokenized=False,
        )
print(f"Tokens: {encoding[0].tokens}")
print(f"Token IDs: {encoding[0].ids}")


input_ids = encoding[0].ids
input_ids = np.array(input_ids,np.int64)
output = model.generate(input_ids,print_progress,tokenizer)



decoded_text = tokenizer.decode(encoding[0].ids)
print(f"Decoded Text: {decoded_text}")


out = output

d_text = tokenizer.decode(out,skip_special_tokens=True)
print("out text")
print(d_text)

# 
print(1 / np.mean(np.array(times)),np.sum(np.array(times)))

# np.save("cache.npy",np.array(model.cacheinput,dtype=object))