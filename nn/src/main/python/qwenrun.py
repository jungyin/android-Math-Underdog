from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers import AddedToken
from tokenizers.processors import TemplateProcessing
import numpy as np

import jinja2
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment
import time

from testrun import QwenMoelRun
from qwen2run import QwenMoelRun

# 读取 tokenizer_config.json 文件
import json

outstrs = ""

t1 = 0
times = []
def print_progress(ntoken,tokenizer,lstr =""):
  """
  调用本函数将打印一个基于当前迭代次数的进度条。
  """
  global t1
  if(t1!=0):
    t2 = time.time() - t1
    times.append(t2)
  t1 =time.time()


  d_text = tokenizer.decode(ntoken[0],skip_special_tokens=True)
  # lstr=lstr+d_text
  # print(d_text, end='\r')
  print(d_text,end='')  # 打印字符但不换行，并立即刷新输出缓冲区

  return lstr

model = QwenMoelRun()

# 配置项
tokenizer_json = './assets/tokenizer.json'
config_path = "./assets/tokenizer_config.json"
add_generation_prompt = True

# 测试数据
prompt= "请帮我写一个傅里叶变化公式,并使用python代码简单复现一下"
testtext =  '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>assistant\n'
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.","user":"yinjun"},
    {"role": "user", "content": prompt,"user":"yinjun"}
]



# 加载 tokenizer_config 文件
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

chat_template = config['chat_template']

def raise_exception(message):
  raise TemplateError(message)

jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
jinja_env.globals["raise_exception"] = raise_exception
compiled_template = jinja_env.from_string(chat_template)


# 加载 tokenizer_config 文件
with open(config_path, 'r', encoding='utf-8') as f:
  config = json.load(f)


rendered=[]

conversations = [messages]

template_kwargs = {
  'eos_token' : config['eos_token'],
  'pad_token' : config['pad_token'],
  'additional_special_tokens':config['additional_special_tokens']
}

for chat in conversations:
  if hasattr(chat, "messages"):
      # Indicates it's a Conversation object
      chat = chat.messages
  rendered_chat = compiled_template.render(
      messages=chat, add_generation_prompt=add_generation_prompt, **template_kwargs
  )
  rendered.append(rendered_chat)


# 前处理一下输入内容
tokenizer = Tokenizer.from_file(tokenizer_json)

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

print()

decoded_text = tokenizer.decode(encoding[0].ids)
print(f"Decoded Text: {decoded_text}")


out = output

d_text = tokenizer.decode(out,skip_special_tokens=True)
print("out text")
print(d_text)

print(1 / np.mean(np.array(times)))