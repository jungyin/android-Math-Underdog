# 伪代码示例：语义相似度

from transformers import BertModel, BertTokenizer
import torch
from torch.nn.functional import cosine_similarity

# 1. 加载预训练的中文BERT模型和分词器 (这里我们只需要基础的BertModel来获取句嵌入)
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertModel.from_pretrained("bert-base-chinese")

# 2. 准备句子对
sentence1 = "今天天气真好"
sentence2 = "今天阳光明媚"
sentence3 = "我肚子饿了"

# 3. 数据预处理
# 将两个句子拼接起来，或者分别编码，这里我们分别编码获取句嵌入
inputs1 = tokenizer(sentence1, return_tensors="pt", padding=True, truncation=True)
inputs2 = tokenizer(sentence2, return_tensors="pt", padding=True, truncation=True)
inputs3 = tokenizer(sentence3, return_tensors="pt", padding=True, truncation=True)

# 4. 获取句嵌入 (通常使用 [CLS] Token的输出作为句子表示)
model.eval()
with torch.no_grad():
    # 获取句子1的 [CLS] Token 向量
    outputs1 = model(**inputs1)
    sentence_embedding1 = outputs1.last_hidden_state[:, 0, :] # [CLS] token vector

    # 获取句子2的 [CLS] Token 向量
    outputs2 = model(**inputs2)
    sentence_embedding2 = outputs2.last_hidden_state[:, 0, :]

    # 获取句子3的 [CLS] Token 向量
    outputs3 = model(**inputs3)
    sentence_embedding3 = outputs3.last_hidden_state[:, 0, :]

# 5. 计算余弦相似度
similarity_1_2 = cosine_similarity(sentence_embedding1, sentence_embedding2).item()
similarity_1_3 = cosine_similarity(sentence_embedding1, sentence_embedding3).item()

print(f"句子1: '{sentence1}'")
print(f"句子2: '{sentence2}'")
print(f"句子3: '{sentence3}'")
print(f"'{sentence1}' 和 '{sentence2}' 的相似度: {similarity_1_2:.4f}")
print(f"'{sentence1}' 和 '{sentence3}' 的相似度: {similarity_1_3:.4f}")