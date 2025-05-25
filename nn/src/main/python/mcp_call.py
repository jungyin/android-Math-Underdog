import json
import requests # 假设用于模拟外部工具调用
from transformers import AutoTokenizer
from infer.qwen.source_infer import QwenMoelRun
from tokenizers import Tokenizer

model_path = "./assets/qewn2/"
model_path =  "D:\\code\\transformer_models\\models--Qwen--Qwen2.5-3B-Instruct/"
model = QwenMoelRun(model_path)
tokenizer_json = model_path+'tokenizer.json'
tokenizer = Tokenizer.from_file(tokenizer_json)


# 模拟 MCP 工具描述 (实际中从 MCP 客户端获取)
MCP_TOOLS_SCHEMA = [
    {
        "name": "get_current_weather",
        "description": "获取指定城市当前的实时天气信息，包括温度、湿度、天气状况等。适用于用户询问天气情况。例如：'北京今天天气怎么样？'",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "需要查询天气的城市名称，例如：北京、上海、London"
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "search_web",
        "description": "使用网络搜索引擎进行实时信息查询。适用于用户询问需要最新数据或广泛信息的问题，例如：'奥林匹克运动会最近一次在哪举行？'，'最新的AI技术有哪些？'",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "需要搜索的关键词或短语"
                }
            },
            "required": ["query"]
            }
    }
    # 更多工具...
]

# 模拟大模型 API 调用
def call_llm(messages, tools_schema):
    # 这是一个简化的大模型调用，实际中会是 HTTP 请求到 OpenAI/Anthropic/Google 等
    # messages 应该包含 System Prompt, User Prompt, Tool Calls, Tool Results 等
    # tools_schema 会被转换成模型能理解的 function/tool descriptions 格式
    
    # 模拟大模型输出
    # print(f"\n--- Calling LLM with messages: {messages} ---")
    
    # 模拟大模型根据输入判断并输出
    # last_user_message = messages[-1]['content'] if messages and messages[-1]['role'] == 'user' else ""
    rendered=[]
    for chat in messages:
        if hasattr(chat, "messages"):
            # Indicates it's a Conversation object
            chat = chat.messages
        rendered_chat = model.compiled_template.render(
            messages=chat, add_generation_prompt=True, **model.template_kwargs
        )
        rendered.append(rendered_chat)

    

    encoding = tokenizer.encode_batch(
            rendered,
            add_special_tokens=True,
            is_pretokenized=False,
        )
    input_ids = encoding[0].ids
    out = model.generate(input_ids,None,tokenizer)
    d_text = tokenizer.decode(out,skip_special_tokens=True)

    return {
        "role": "assistant",
        "content": d_text
    }

    if "天气" in last_user_message and "北京" in last_user_message:
        return {
            "role": "assistant",
            "content": "<tool_code>\n{\"tool_name\": \"get_current_weather\", \"parameters\": {\"location\": \"北京\"}}\n</tool_code>"
        }
    elif "最新的AI技术" in last_user_message:
        return {
            "role": "assistant",
            "content": "<tool_code>\n{\"tool_name\": \"search_web\", \"parameters\": {\"query\": \"最新的AI技术\"}}\n</tool_code>"
        }
    elif "25C" in last_user_message or "晴朗" in last_user_message: # 假设这是工具返回后的上下文
        return {
            "role": "assistant",
            "content": "好的，根据查询结果，北京今天的天气晴朗，温度25摄氏度。"
        }
    elif "大模型" in last_user_message and "RAG" in last_user_message:
        return {
            "role": "assistant",
            "content": "RAG（检索增强生成）是一种技术，它通过从外部知识库检索相关信息来增强大模型的回答，解决知识滞后和幻觉问题。它通常在生成回答之前进行检索。"
        }
    else:
        return {
            "role": "assistant",
            "content": "抱歉，我目前无法回答这个问题，或者我没有合适的工具来处理您的请求。"
        }

# 模拟 MCP 服务器端的功能执行
def execute_tool(tool_name, parameters):
    print(f"\n--- Executing Tool: {tool_name} with params: {parameters} ---")
    if tool_name == "get_current_weather":
        location = parameters.get("location")
        if location == "北京":
            return {"temperature": "25C", "condition": "晴朗", "humidity": "60%"}
        else:
            return {"error": f"无法获取 {location} 的天气信息"}
    elif tool_name == "search_web":
        query = parameters.get("query")
        # 模拟网络搜索结果
        return {"results": [f"关于'{query}'的最新信息：LLM Agent, MoE, Gemini 2等", "AI安全与伦理研究进展"]}
    else:
        return {"error": f"不支持的工具: {tool_name}"}

# AI Agent 的核心循环 (多步推理)
def run_ai_agent(user_query):
    conversation_history = []
    
    # 1. 初始 System Prompt (包含工具描述)
    system_prompt = {
        "role": "system",
        "content": f"""你是一个功能强大的AI助手，能够理解并执行复杂任务。
        你的核心职责是：
        1.  **根据用户请求，决定是否需要使用外部工具。**
        2.  **如果需要，请严格按照以下格式输出工具调用指令。在输出工具指令前，请先思考你为什么要调用此工具。**
        3.  **在工具执行完成后，你会收到工具的输出。请根据工具输出的内容，继续完成任务或直接回答用户。**
        **工具调用格式示例 (JSON 格式):**
        <tool_code>
        {{
          "tool_name": "工具名称",
          "parameters": {{"参数名1": "值1", "参数名2": "值2"}}
        }}
        </tool_code>

        **可用工具：**
        {json.dumps(MCP_TOOLS_SCHEMA, indent=2, ensure_ascii=False)}
        """
    }
    conversation_history.append(system_prompt)

    # 2. 添加用户初始查询
    user_message = {"role": "user", "content": user_query}
    conversation_history.append(user_message)

    max_turns = 5 # 限制最大对话轮次，防止无限循环
    current_turn = 0

    while current_turn < max_turns:
        current_turn += 1
        print(f"\n--- Turn {current_turn} ---")
        
        # 3. 调用大模型获取响应
        llm_response = call_llm([conversation_history], MCP_TOOLS_SCHEMA)
        
        response_content = llm_response.get("content", "")
        print(f"LLM Response: {response_content}")

        # 4. 判断是否是工具调用
        if response_content.startswith("<tool_code>") and response_content.endswith("</tool_code>"):
            try:
                # 解析工具调用指令
                tool_call_str = response_content.replace("<tool_code>", "").replace("</tool_code>", "").strip()
                tool_call_data = json.loads(tool_call_str)
                tool_name = tool_call_data["tool_name"]
                parameters = tool_call_data.get("parameters", {})

                # 将大模型的工具调用输出添加到对话历史
                conversation_history.append({"role": "assistant", "content": response_content})

                # 5. 执行工具
                tool_result = execute_tool(tool_name, parameters)
                
                # 6. 将工具结果作为上下文回传给大模型
                tool_result_message = {
                    "role": "tool", # 或者 'function' role in OpenAI's API
                    "name": tool_name,
                    "content": json.dumps(tool_result, ensure_ascii=False)
                }
                conversation_history.append(tool_result_message)
                
                # 再次循环，让大模型根据工具结果生成回答
                continue # 进入下一轮循环，LLM会看到Tool Result并继续生成
                
            except json.JSONDecodeError:
                print("Error: LLM returned invalid tool_code JSON.")
                conversation_history.append({"role": "tool", "name": "error", "content": "LLM returned invalid tool_code JSON."})
                break # 终止循环
            except KeyError:
                print("Error: LLM returned malformed tool_code (missing name/parameters).")
                conversation_history.append({"role": "tool", "name": "error", "content": "LLM returned malformed tool_code."})
                break # 终止循环
        else:
            # 如果不是工具调用，说明大模型直接给出了最终答案或无法处理
            print(f"Final Answer: {response_content}")
            conversation_history.append({"role": "assistant", "content": response_content})
            break # 任务完成，退出循环

    if current_turn >= max_turns:
        print("Max turns reached. Task might not be fully completed.")
    
    return conversation_history[-1].get("content", "No final answer.")

if __name__ == "__main__":
    # print("\n--- Scenario 1: Weather Query ---")
    # final_response_1 = run_ai_agent("北京明天天气怎么样？适合出去旅游吗")
    # print(f"\nUser query: '北京今天天气怎么样？'")
    # print(f"Agent's final response: {final_response_1}")

    # print("\n\n--- Scenario 2: Web Search ---")
    # final_response_2 = run_ai_agent("最新的AI技术有哪些？")
    # print(f"\nUser query: '最新的AI技术有哪些？'")
    # print(f"Agent's final response: {final_response_2}")

    # print("\n\n--- Scenario 3: Direct Answer ---")
    # final_response_3 = run_ai_agent("RAG 和多步推理有什么区别？")
    # print(f"\nUser query: 'RAG 和多步推理有什么区别？'")
    # print(f"Agent's final response: {final_response_3}")

    # print("\n\n--- Scenario 4: Unhandled Query ---")
    # final_response_4 = run_ai_agent("请帮我订一张明天去上海的机票。")
    # print(f"\nUser query: '请帮我订一张明天去上海的机票。'")
    # print(f"Agent's final response: {final_response_4}")