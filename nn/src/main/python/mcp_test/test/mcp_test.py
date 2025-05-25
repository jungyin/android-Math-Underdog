import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from modelcontextprotocol.client import MCPClient
from fastapi import FastAPI
import uvicorn

# ==================== 第一步：加载本地模型 ====================
# 假设你有一个本地训练好的模型（如 Llama-3 或 Qwen）
model_name = "Qwen/Qwen2-7B"  # 替换为你的本地模型路径或 HuggingFace 模型名

model_f = "D:\\code\\transformer_models\\"
model = "models--Qwen--Qwen2.5-3B-Instruct"
model = "models--Qwen--Qwen2.5-0.5B-Instruct"
model_path = model_f + model

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)

# ==================== 第二步：初始化 MCP 客户端 ====================
# 假设 MCP 服务已部署在本地（例如天气查询服务）
mcp_client = MCPClient(base_url="http://localhost:8080")  # MCP 服务地址

# ==================== 第三步：定义 MCP 工具调用逻辑 ====================
def handle_user_query(query: str):
    # 1. 使用模型生成初步回复（可能包含需要调用 MCP 的指令）
    model_response = text_generator(query, return_full_text=False)[0]["generated_text"]
    
    # 2. 检查是否需要调用 MCP 服务（示例逻辑：检测关键词）
    if "天气" in query:
        city = extract_city(query)  # 提取城市名
        weather_data = call_mcp_weather(city)  # 调用 MCP 服务
        model_response += f"\n\n根据天气数据：{city} 当前温度 {weather_data['temperature']}°C，天气 {weather_data['condition']}"
    return model_response

def extract_city(query: str) -> str:
    # 简单提取城市名（实际可用 NER 模型）
    cities = ["北京", "上海", "广州", "深圳", "杭州"]
    for city in cities:
        if city in query:
            return city
    return "北京"

def call_mcp_weather(city: str) -> dict:
    # 调用 MCP 服务（假设服务名为 "weather"，方法为 "get_weather"）
    response = mcp_client.call("weather", "get_weather", {"city": city})
    return response

# ==================== 第四步：创建 FastAPI 接口（可选） ====================
app = FastAPI()

@app.post("/chat")
async def chat_endpoint(query: str):
    return {"response": handle_user_query(query)}

if __name__ == "__main__":
    # 启动本地服务（测试用）
    uvicorn.run(app, host="0.0.0.0", port=5000)


