# 伪代码示例：零样本分类（概念性）

from transformers import BertModel, BertTokenizer
import torch
from torch.nn.functional import cosine_similarity

# 1. 加载预训练的中文BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertModel.from_pretrained("bert-base-chinese")

# 2. 准备待分类文本和类别描述
text_to_classify = "这个产品功能强大，非常实用！"
# 定义类别标签的语义描述，这比直接的类别名称包含更多信息
class_descriptions = {
    "正面": "这是一条关于产品正面评价的文本。",
    "负面": "这是一条关于产品负面评价的文本。",
    "中立": "这是一条关于产品中立评价的文本。"
}

# 3. 获取文本和所有类别描述的嵌入
all_texts = [text_to_classify] + list(class_descriptions.values())
inputs = tokenizer(all_texts, return_tensors="pt", padding=True, truncation=True)

# 4. 获取所有文本的 [CLS] Token 嵌入
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :] # [CLS] embeddings

text_embedding = embeddings[0:1] # 待分类文本的嵌入
description_embeddings = embeddings[1:] # 类别描述的嵌入

# 5. 计算待分类文本与每个类别描述的相似度
similarities = []
for i, (label, desc) in enumerate(class_descriptions.items()):
    sim = cosine_similarity(text_embedding, description_embeddings[i:i+1]).item()
    similarities.append((label, sim))

# 6. 选择相似度最高的类别作为预测结果
predicted_class = max(similarities, key=lambda x: x[1])

print(f"待分类文本: '{text_to_classify}'")
print("\n零样本分类结果:")
for label, sim in similarities:
    print(f" - 与 '{label}' 类别描述的相似度: {sim:.4f}")
print(f"\n预测类别: {predicted_class[0]} (相似度: {predicted_class[1]:.4f})")