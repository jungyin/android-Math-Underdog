import requests
from transformers import pipeline

# 假设你的MCP服务器地址是 http://localhost:8000/mcp_endpoint
MCP_SERVER_URL = "http://localhost:8000/mcp_endpoint"

def call_mcp_server(data):
    """调用MCP服务器并返回结果"""
    response = requests.post(MCP_SERVER_URL, json=data)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error calling MCP server: {response.text}")

# 创建一个文本分类pipeline作为示例
classifier = pipeline("text-classification")

def enhanced_pipeline(text):
    # 首先调用原始的Transformer模型pipeline
    original_result = classifier(text)

    # 准备发送给MCP服务器的数据
    data_to_send = {"text": text, "original_result": original_result}

    # 调用MCP服务器获取额外的信息
    mcp_response = call_mcp_server(data_to_send)

    # 结合原始结果和从MCP服务器得到的结果
    combined_result = {
        "original_classification": original_result,
        "mcp_enhancements": mcp_response
    }

    return combined_result

# 示例使用
if __name__ == "__main__":
    text = "这是一个测试句子。"
    result = enhanced_pipeline(text)
    print(result)