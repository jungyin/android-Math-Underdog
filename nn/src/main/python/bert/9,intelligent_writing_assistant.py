# 伪代码示例：文本补全 (基于Masked Language Model)

from transformers import BertForMaskedLM, BertTokenizer
import torch

# 1. 加载预训练的中文BERT Masked Language Model 和分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertForMaskedLM.from_pretrained("bert-base-chinese")

# 2. 准备带有 [MASK] 标记的文本
text_to_complete = "我爱北京[MASK]大学。"
# 找到 [MASK] token 的 ID
mask_token_id = tokenizer.mask_token_id
mask_token_index = tokenizer.encode(text_to_complete, add_special_tokens=True).index(mask_token_id)

# 3. 编码输入
inputs = tokenizer(text_to_complete, return_tensors="pt")

# 4. 进行预测
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits # 预测每个 [MASK] 位置的词汇概率

# 5. 获取 [MASK] 位置的最高概率词汇
predicted_token_logits = predictions[0, mask_token_index, :]
top_k_tokens = torch.topk(predicted_token_logits, 5).indices.tolist()

print(f"原始文本: {text_to_complete}")
print("可能的补全词汇:")
for token_id in top_k_tokens:
    token = tokenizer.decode([token_id])
    print(f" - {token}")

# 替换 [MASK] 并打印一个示例
completed_text = text_to_complete.replace("[MASK]", tokenizer.decode([top_k_tokens[0]]))
print(f"\n最佳补全示例: {completed_text}")