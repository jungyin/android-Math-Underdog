from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers import AddedToken
from tokenizers.processors import TemplateProcessing
import numpy as np

import jinja2
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment
      


# 读取 tokenizer_config.json 文件
import json

# 预处理 qwen2的输入数据
def prepare_inputs_for_generation(
  self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None,seen_tokens=51, **kwargs
):

  past_length = seen_tokens
   # Keep only the unprocessed tokens:
  # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
  # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
  # input)
  if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
      input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
  # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
  # input_ids based on the past_length.
  elif past_length < input_ids.shape[1]:
      input_ids = input_ids[:, past_length:]

      
  if attention_mask is not None and position_ids is None:
    # create position_ids on the fly for batch generation
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    if past_key_values:
        position_ids = position_ids[:, -input_ids.shape[1] :]

  # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
  if inputs_embeds is not None and past_key_values is None:
      model_inputs = {"inputs_embeds": inputs_embeds}
  else:
      model_inputs = {"input_ids": input_ids}

  model_inputs.update(
      {
          "position_ids": position_ids,
          "past_key_values": past_key_values,
          "use_cache": kwargs.get("use_cache"),
          "attention_mask": attention_mask,
      }
  )
  return model_inputs



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




decoded_text = tokenizer.decode(encoding[0].ids)
print(f"Decoded Text: {decoded_text}")


out = [102645,  69249, 100027, 114714, 101158,  44063, 104757,  45181,  39973,
          8863,  99390, 105359,  17714,  58364,   8863,  99390, 104339,   3837,
         99652, 104193, 104757, 100629,  61149,  33071,   1773, 102645,  69249,
        100027, 114714, 110322,  17714,  48443,  78045,    282,   1155,      8,
           284,   1124,   1242,  15159,     77,     28,     15,     92,     61,
         35702,    258,  36958,     92,    272,   1089,    384,  47822,     72,
         69761,   9139,   1124,   2533,  90919,  17767,    272,   1089,   1124,
             8,  54851, 110589,   3837,  44292,    308,   1124,      8,  54851,
        107586,   1773, 100431, 101909, 105172,  30280,  46100,  19793,  26355,
          3837,  37029,     63,  35083,     63,  44956,  36407, 101884, 102645,
         69249, 100027, 114714,   3407,  73594,  12669,    198,    474,   8591,
           438,   2595,    271,      2,  41479,    248,  64559,  46944, 104757,
           198,     83,    284,   2595,  38712,      7,     15,     11,    220,
            17,    353,   2595,  24259,     11,    220,     16,     15,     15,
            15,    340,  26622,    284,   2595,  16318,   1155,    692,      2,
         33424,     94,  69103, 102645,  69249, 100027, 114714,    198,     69,
           284,   2595,  79899,  79899,  56782,    692,      2,    220,  46485,
        102645,  69249, 100027, 114714,   9370, 110589,    198,  48638,  28142,
           284,    282,     58,     15,   2533,   1350,  67018,  28142,    340,
         13874,  19324, 104596,  19793,  26355,  15946,   3837,  97639, 101140,
         91282, 104059, 102298,  20450,  44292,    259,   1124,      8,   9370,
         69824,  90395,  50377, 104059, 102298,  36556, 106514, 104757,   9370,
         69824,  44292,   8286,   1124,      8,   1773, 101889,   3837,  97639,
         37029,     63,   6199,  79899,  79899,     63,  32804, 100768, 102645,
         69249, 100027, 114714,   1773, 100161,   3837,  97639, 107439, 102645,
         69249, 100027, 114714,   9370, 110589,  62926, 102703,  99898,   3407,
        104001, 107083,  46100,   3837, 102762, 101051,  46944, 102298,  64952,
        101454, 110589,   9370,  69824,   3837, 103991, 102268,  99661, 106168,
        107586,   9370, 102645,  69249, 100027, 114714, 110589,   1773, 151645]

out = np.array(out)

d_text = tokenizer.decode(out,skip_special_tokens=True)
print(d_text)


