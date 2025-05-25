import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # 初始化pycuda

# 设置日志级别
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path):
    """构建TensorRT引擎"""
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:
        
        builder.max_workspace_size = 1 << 30  # 1GB
        builder.max_batch_size = 1
        
        # 可根据需要调整精度
        # builder.fp16_mode = True
        
        # 读取ONNX文件
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        # 构建TensorRT引擎
        engine = builder.build_cuda_engine(network)
        return engine

def serialize_engine(engine, engine_file_path):
    """序列化TensorRT引擎到磁盘"""
    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())

def deserialize_engine(engine_file_path):
    """从磁盘反序列化TensorRT引擎"""
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def infer(engine, inputs):
    """执行推理"""
    context = engine.create_execution_context()
    
    # 分配显存
    d_inputs = [cuda.mem_alloc(input.nbytes) for input in inputs]
    output = np.empty((1, 1000), dtype=np.float32)  # 假设输出大小为1x1000
    d_output = cuda.mem_alloc(output.nbytes)
    
    # 将输入数据复制到GPU
    for h_input, d_input in zip(inputs, d_inputs):
        cuda.memcpy_htod(d_input, h_input)
    
    # 执行推理
    bindings = [int(d_input) for d_input in d_inputs] + [int(d_output)]
    context.execute_v2(bindings)
    
    # 将结果复制回主机内存
    cuda.memcpy_dtoh(output, d_output)
    
    return output

# 示例使用
# onnx_file_path = 'model.onnx'
engine_file_path = './model_file/qwen2-code-0.5b/tensorrt/model32.trt'

# 构建并序列化引擎
# engine = build_engine(onnx_file_path)
# if engine is not None:
    # serialize_engine(engine, engine_file_path)

# 反序列化引擎
engine = deserialize_engine(engine_file_path)

# 准备输入数据
input_ids = np.random.random(size=(1,47)).astype(np.float32)
attention_mask = np.zeros([1,47]).astype(np.float32)

position_ids = attention_mask.long().cumsum(-1) - 1
position_ids.masked_fill_(attention_mask == 0, 1)
past_key_value = np.random.random(size=(24, 2, 1, 2, 0, 64)).astype(np.float32)

# 执行推理
output = infer(engine, [input_ids,attention_mask,position_ids,past_key_value])
print(output)