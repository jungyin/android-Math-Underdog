# 伪代码示例：情感分析

from transformers import BertForSequenceClassification, BertTokenizer
import torch

# 1. 加载预训练的中文BERT模型和分词器
# BertForSequenceClassification 已经包含了BERT模型和一个分类头
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
# num_labels=3 表示有三个情感类别：正面、负面、中立
model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=3)

# 2. 准备数据 (假设有评论文本和对应的情感标签)
# 实际中，你需要加载你的数据集
train_texts = ["这部电影太棒了！", "完全是浪费时间。", "还行吧，没什么特别的。"]
train_labels = [0, 1, 2] # 0:正面, 1:负面, 2:中立

# 3. 数据预处理 (Tokenization和转换为模型输入格式)
encoded_inputs = tokenizer(train_texts, padding=True, truncation=True, return_tensors="pt")
input_ids = encoded_inputs["input_ids"]
attention_mask = encoded_inputs["attention_mask"]
labels = torch.tensor(train_labels)

# 4. 模型微调 (训练)
# 这是一个简化的训练循环概念
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
model.train() # 设置模型为训练模式

for epoch in range(3): # 迭代几个epochs
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward() # 反向传播
    optimizer.step() # 更新模型参数
    optimizer.zero_grad() # 清空梯度
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# 5. 进行预测 (推断)
model.eval() # 设置模型为评估模式
test_text = "电影情节很紧张刺激！"
test_inputs = tokenizer(test_text, return_tensors="pt")

with torch.no_grad(): # 不计算梯度，节省内存和计算
    outputs = model(**test_inputs)
    logits = outputs.logits # 获取模型的原始输出 (logits)
    predicted_class_id = torch.argmax(logits, dim=1).item()

# 将类别ID映射回标签 (根据你的定义)
id_to_label = {0: "正面", 1: "负面", 2: "中立"}
print(f"评论: '{test_text}' -> 情感: {id_to_label[predicted_class_id]}")