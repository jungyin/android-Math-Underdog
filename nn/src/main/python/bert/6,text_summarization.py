# 伪代码示例：抽取式摘要（概念性）

from transformers import BertModel, BertTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity

# 1. 加载预训练的中文BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertModel.from_pretrained("bert-base-chinese")

# 2. 准备长篇文本
long_text = "中国经济在过去几十年取得了显著的增长，成为全球第二大经济体。上海和北京等大都市在金融、科技和文化方面发挥着重要作用。政府持续推动创新和可持续发展，以应对全球挑战。"

# 3. 将文本分割成句子
# 实际中需要更复杂的中文句子分割工具，这里简化
sentences = long_text.split("。")
sentences = [s.strip() + "。" for s in sentences if s.strip()] # 确保句号完整

# 4. 获取每个句子的嵌入向量
sentence_embeddings = []
model.eval()
with torch.no_grad():
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        # 通常使用 [CLS] token 的输出作为句子嵌入
        sentence_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        sentence_embeddings.append(sentence_embedding)

# 5. 简单地选择与整个文档“最相似”的句子作为摘要（非常简化的抽取逻辑）
# 更复杂的抽取式摘要会涉及TextRank等算法
# 首先，计算整个文档的平均嵌入（代表文档主旨）
if sentence_embeddings:
    doc_embedding = torch.mean(torch.tensor(sentence_embeddings), dim=0).unsqueeze(0).numpy()
    
    # 计算每个句子与文档平均嵌入的相似度
    similarities = []
    for i, emb in enumerate(sentence_embeddings):
        sim = cosine_similarity(doc_embedding, emb.reshape(1, -1))[0][0]
        similarities.append((sim, i)) # (相似度, 句子索引)

    # 按相似度降序排序，选择最相似的N个句子
    similarities.sort(key=lambda x: x[0], reverse=True)

    # 假设我们选择前2个句子作为摘要
    num_summary_sentences = min(2, len(sentences))
    summary_indices = sorted([s[1] for s in similarities[:num_summary_sentences]])
    extracted_summary = [sentences[i] for i in summary_indices]

    print(f"原文:\n{long_text}\n")
    print("抽取式摘要 (简化示例):")
    for s in extracted_summary:
        print(s)
else:
    print("文本无法有效分割或生成摘要。")