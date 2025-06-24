import onnx
import collections

# 加载 ONNX 模型

from collections import defaultdict
import onnx

def load_onnx_model_structure(model_path):
    model_proto = onnx.ModelProto()
    with open(model_path, 'rb') as f:
        model_proto.ParseFromString(f.read())
    return model_proto.graph
# 自定义分类器
def classify_module(node):
    name = node.name.lower()
    if "embed" in name:
        return "Embedding"
    elif "rmsnorm" in name:
        return "RMSNorm"
    elif "pos" in name or "position" in name:
        return "PositionalEncoding"
    elif "decoder" in name or ("attn" in name and "layer" in name):
        return "DecoderLayer"
    else:
        return "Other"

def analyze_modules(graph):
    module_info = defaultdict(lambda: {"count": 0, "parents": set()})
    
    for node in graph.node:
        module_type = classify_module(node)
        
        # 获取父节点名称集合
        parents = {input_node for input_node in node.input}
        
        # 更新计数和父节点信息
        module_info[module_type]["count"] += 1
        module_info[module_type]["parents"].update(parents)
    
    return module_info

def print_analysis(module_info):
    for module, info in module_info.items():
        print(f"{module}: 出现次数={info['count']}, 父节点={info['parents']}")
        # print(f"{module}: 出现次数={info['count']}")5

if __name__ == "__main__":
    # model_path = "your_llama_model.onnx"
        
    model_path = "D:/code/py/qwen2demo_py/onnx/coder-0.5bfp32/model32.onnx"
    graph = load_onnx_model_structure(model_path)  # 假设此函数已定义
    module_info = analyze_modules(graph)
    print_analysis(module_info)