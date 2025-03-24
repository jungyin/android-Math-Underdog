import torch
import torch.nn as nn
import math

# 1. 定义 Positional Encoding 层
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len, _ = x.size()
        x = x + self.pe[:, :seq_len, :]
        return x

# 2. 定义网络模型
class PositionalEncodingNetwork(nn.Module):
    def __init__(self, input_dim, d_model, output_dim):
        super(PositionalEncodingNetwork, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)  # 输入嵌入层
        self.positional_encoding = PositionalEncoding(d_model=d_model)  # 位置编码层
        self.fc = nn.Linear(d_model, output_dim)  # 输出层

    def forward(self, x):
        x = self.embedding(x)  # 嵌入
        x = self.positional_encoding(x)  # 添加位置编码
        x = torch.mean(x, dim=1)  # 对序列维度进行平均池化
        x = self.fc(x)  # 输出层
        return x

# 3. 创建模型实例
input_dim = 10  # 输入特征维度
d_model = 64    # 嵌入维度
output_dim = 5  # 输出维度

model = PositionalEncodingNetwork(input_dim=input_dim, d_model=d_model, output_dim=output_dim)
model.half()

# 4. 定义输入张量
dummy_input = torch.randn(1, 8, input_dim,dtype=torch.float16)  # (batch_size, seq_len, input_dim)

# 5. 导出为 ONNX 模型
onnx_path = "positional_encoding_network.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    opset_version=13,  # 使用 Opset 13 或更高版本以支持更多算子
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size", 1: "seq_len"}, "output": {0: "batch_size"}}
)

print(f"模型已成功导出为 ONNX 格式，文件路径: {onnx_path}")