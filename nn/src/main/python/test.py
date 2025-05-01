from transformers import AutoTokenizer
import torch
# 初始化一个预训练的BERT tokenizer

import onnx_tool


model_path = 'D:\code\py\qwen\onnx\qwen2_3b_fp16\model.onnx' #本地模型

onnx_tool.model_profile(model_path)

model_id = "Qwen/Qwen2.5-Coder-3B-Instruct"
model_id = "Qwen/Qwen2.5-3B"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_default_system_prompt=True)

# 定义最大输入长度
max_length = 512

# 示例文本
text = "你的示例文本"

# 编码文本，设置最大长度并进行填充或截断
encoding = tokenizer.encode_plus(
    text,
    add_special_tokens=True,  # 添加 [CLS] 和 [SEP] 标记
    max_length=max_length,
    padding='max_length',  # 填充到最大长度
    return_attention_mask=True,  # 返回 attention mask
    return_tensors='pt',  # 返回 PyTorch 张量
    truncation=True  # 如果超过最大长度则截断
)

# 获取 input_ids, attention_mask
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']

# 对于BERT模型，位置编码是自动处理的，但如果你想手动创建它们
position_ids = torch.arange(0, max_length, dtype=torch.long).unsqueeze(0)  # 形状 (1, max_length)

print(f"Input IDs shape: {input_ids.shape}")
print(f"Attention Mask shape: {attention_mask.shape}")
print(f"Position IDs shape: {position_ids.shape}")