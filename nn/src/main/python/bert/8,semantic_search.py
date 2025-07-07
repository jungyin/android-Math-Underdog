# 伪代码示例：语义搜索

from transformers import BertModel, BertTokenizer
import torch
from torch.nn.functional import cosine_similarity

# 1. 加载预训练的中文BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertModel.from_pretrained("bert-base-chinese")

# 2. 准备文档库
documents = [
    "北京是中国的首都，拥有故宫和长城等著名景点。",
    "上海的金融中心地位日益突出，是国际商业枢纽,中国的金融中心",
    "今天天气晴朗，适合外出游玩。",
    "上海是中国的金融中心"
    "中国的金融中心是上海"
    "人工智能技术正在改变我们的生活，应用于各个领域。"
]

# 3. 预先计算所有文档的嵌入 (实际中会离线完成并存储)
document_embeddings = []
model.eval()
with torch.no_grad():
    for doc in documents:
        inputs = tokenizer(doc, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        doc_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        document_embeddings.append(doc_embedding)

document_embeddings = torch.tensor(document_embeddings)

# 4. 用户查询
query = "寻找有关中国金融中心的资料"

# 5. 计算查询的嵌入
query_inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    query_outputs = model(**query_inputs)
    query_embedding = query_outputs.last_hidden_state[:, 0, :].squeeze()

# 6. 计算查询嵌入与所有文档嵌入的相似度
similarities = cosine_similarity(query_embedding.unsqueeze(0), document_embeddings)

# 7. 排序并返回最相关的文档
ranked_docs_indices = torch.argsort(similarities, descending=True)

print(f"用户查询: '{query}'")
print("\n语义搜索结果 (按相关性排序):")
for i in ranked_docs_indices:
    doc_index = i.item()
    print(f"  - 文档: '{documents[doc_index]}' (相似度: {similarities[doc_index].item():.4f})")