import torch
import os

try:
    # 尝试从 flash_attn.flash_attn_interface 导入 flash_attn_func
    # 这是 flash-attention 2.x 版本常用的导入路径
    from flash_attn.flash_attn_interface import flash_attn_func
    print("成功从 flash_attn.flash_attn_interface 导入 flash_attn_func。")
except ImportError:
    try:
        # 如果上面失败，尝试从 flash_attn.flash_attention 导入 flash_attn_cuda
        # 这是 flash-attention 1.x 版本可能使用的导入路径
        from flash_attn.flash_attention import flash_attn_cuda
        flash_attn_func = flash_attn_cuda # 统一函数名
        print("成功从 flash_attn.flash_attention 导入 flash_attn_cuda。")
    except ImportError:
        print("错误：无法导入 flash_attn 模块。请确认是否已正确安装 Flash-Attention。")
        print("你可能需要运行: pip install flash-attn --no-build-isolation")
        exit()

print("-" * 30)
print("开始 Flash-Attention 安装验证...")

# 1. 检查 CUDA 是否可用
if not torch.cuda.is_available():
    print("错误：未检测到 CUDA 可用。Flash-Attention 需要 GPU。")
    print("请检查你的 PyTorch 安装是否支持 CUDA，并确保 GPU 驱动正确。")
    exit()

device = "cuda"
dtype = torch.float16 # Flash-Attention 通常使用 FP16 或 BF16
print(f"CUDA 可用，将使用设备：{device}，数据类型：{dtype}")

# 2. 定义注意力机制的输入维度 (随机权重，用于测试)
batch_size = 2
num_heads = 8
seqlen_q = 128 # query 序列长度
seqlen_kv = 128 # key/value 序列长度
head_dim = 64 # 每个头的维度

print(f"输入参数：Batch={batch_size}, Heads={num_heads}, Q_SeqLen={seqlen_q}, KV_SeqLen={seqlen_kv}, Head_Dim={head_dim}")

# 3. 创建随机输入张量并移动到 GPU
try:
    q = torch.randn(batch_size, seqlen_q, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen_kv, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen_kv, num_heads, head_dim, device=device, dtype=dtype)
    print("成功创建随机输入张量并移至 GPU。")
except Exception as e:
    print(f"错误：创建张量或移动到 GPU 失败：{e}")
    exit()

# 4. 调用 Flash-Attention 前向函数
print("正在调用 flash_attn_func 进行前向传递...")
try:
    # is_causal=False 表示非因果注意力，可以设置为 True 测试因果模式
    output = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)
    print("Flash-Attention 前向传递成功完成！")

    print(f"输出张量形状：{output.shape}")
    print(f"输出张量设备：{output.device}")
    print(f"输出张量数据类型：{output.dtype}")

    # 可以进行一个简单的数值检查，确保不是 NaN 或 Inf
    if torch.isnan(output).any() or torch.isinf(output).any():
        print("警告：输出中包含 NaN 或 Inf 值，这可能表示输入过大或数值不稳定。")

    print("\nFlash-Attention 安装验证通过！")

except Exception as e:
    print(f"错误：调用 flash_attn_func 失败：{e}")
    print("这可能意味着：")
    print("1. Flash-Attention 未正确编译或安装。")
    print("2. CUDA 版本、PyTorch 版本与 Flash-Attention 不兼容。")
    print("3. GPU 内存不足（对于更大的输入）。")
    exit()

print("-" * 30)