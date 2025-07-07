# 伪代码示例：问答系统 (阅读理解型)

from transformers import BertForQuestionAnswering, BertTokenizer
import torch

# 1. 加载预训练的中文BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertForQuestionAnswering.from_pretrained("bert-base-chinese")

# 2. 准备输入：上下文 (文章) 和 问题
context = "上海是一个国际化大都市，拥有众多金融机构和繁华的商业区，是中国经济的中心之一。"
question = "上海是中国哪个方面的中心？"

# 3. 数据预处理
inputs = tokenizer(question, context, return_tensors="pt", truncation=True, padding=True)

# 4. 进行预测
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    # outputs.start_logits 预测每个Token作为答案起始点的概率
    # outputs.end_logits 预测每个Token作为答案结束点的概率
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

# 5. 找到概率最高的起始和结束点
answer_start = torch.argmax(answer_start_scores)
answer_end = torch.argmax(answer_end_scores) + 1 # 结束点通常是独占的，所以加1

# 6. 从原始文本中提取答案
input_ids = inputs["input_ids"].squeeze().tolist()
answer_tokens = tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
answer = tokenizer.convert_tokens_to_string(answer_tokens)

print(f"上下文: {context}")
print(f"问题: {question}")
print(f"答案: {answer}")