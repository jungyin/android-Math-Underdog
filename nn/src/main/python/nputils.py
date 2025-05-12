import numpy as np
# numpy 的工具集，用于实现numpy不支持，但想要模仿pytorch的方法

def softmax(x,axis=-1):


    # 计算输入向量的最大值，并保持其原始维度
    x_max = np.max(x, axis=axis, keepdims=True)
    # 对平移后的向量进行指数运算
    e_x = np.exp(x - x_max)
    # 计算指数运算结果的和，并保持其原始维度
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multinomial_numpy(probs, num_samples=1):
    # 获取批次大小和词汇表大小
    batch_size, vocab_size = probs.shape
    # 初始化结果数组
    next_tokens = np.zeros(batch_size, dtype=np.int64)
    # 对每个样本单独采样
    for i in range(batch_size):
        # 使用 numpy.random.choice 按照概率分布采样
        work_num =  np.random.choice(vocab_size, size=num_samples, p=probs[i], replace=True)
        next_tokens[i] =work_num[0]
    return next_tokens.squeeze()  # 如果 num_samples=1，则去掉多余的维度

def masked_fill (x,mask,full_value):
    x[mask] = full_value
    return x