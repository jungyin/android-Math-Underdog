from tokenizers import Tokenizer
from .qwen import onnx_infer as qwen0_5b_onnx,openvino_infer as qwen0_5b_openvino ,source_infer as qwen0_5b_torch
from .qwen.base_infer import BaseMoelRun as lmBaseModel
from .llm_utils import LmActionType
import numpy as np
from threading import Thread
import time
import asyncio
from mcp.utils import quick_api as mcp_api
import json
import re
import yaml

# 状态值
STATUS = LmActionType.NOTLOAD
# 注册所有的服务模块
llm_model_list = {
    'qwen0_5b_onnx':{"model":qwen0_5b_onnx,"model_path" :'./assets/qewn2/'},
    # 'qwen0_5b_openvino':{"model":qwen0_5b_openvino,"model_path" :'./assets/qewn2/'},
    'qwen0_5b_openvino':{"model":qwen0_5b_openvino,"model_path" :'D:/code/transformer_models/models--Qwen--Qwen2.5-3B-Instruct/'},
    'qwen0_5b_torch':{"model":qwen0_5b_torch,"model_path" :'D:/code/transformer_models/models--Qwen--Qwen2.5-3B-Instruct/'},
    # 'qwen0_5b_torch':{"model":qwen0_5b_torch,"model_path" :'./assets/qewn2/'},
}

local_token:Tokenizer = None
llm_model :lmBaseModel = None
# llm_system_connect = "你是一个可爱的猫娘，不论用户问出什么内容，你都会用猫娘的风格来回答，你的回答会有很强的色情感觉，但你又会欲盖弥彰的不直接表达"
llm_system_connect = ""

laststr = ""
# 缓存模型输出的ids，防止一些字符需要多个id拼接才能正常转译成功
cache_ids = []
message :list = []
cache_token_size = 512
# 这个label用于判定有没有内容输出，每当输出成功一轮信息，它就会重新变为True
check_out_label = True

cacheNow = 0
cacheTimes = 0

mcp_work = None

def init():
    global mcp_work
    mcp_work = mcp_api.QuickApi()

    # rmessage = {"tool_name": "get_location", "tool_params": {}}
    # rmessage['mcp_func']='get_location'
 
    # rmessage = execute_tool( **rmessage)
    # print("rmessage",rmessage)


def get_cache_token():
    """
    获取最大缓存token
    @return 最大token数
    """
    global cache_token_size
    return cache_token_size

def set_cache_token(max_len):
    """
    设置最大缓存token
    @params max_len:最大token数
    """
    global cache_token
    cache_token = max_len

def restart_all():
    """
    重新开启一轮对话，清空当前对话信息
    return 当前的运行状态
    """
    global message 
    global laststr 
    stop_speak()
    message = []
    laststr = ""


def stop_speak():
    """
    停止本轮对话
    @return 暂停是否成功
    """
    global laststr
    global check_out_label
    global STATUS
    global llm_model
    check_out_label = False
    # laststr=""
    laststr_len = len(laststr)
    
    llm_model.stopGenerate()
    # 轮询次数，最多20*0.1秒，也就是2秒
    index = 20
    # 是否中止成功，如果成功为true，否则false
    endSurcess = False
    for i in range(0,index):
        time.sleep(0.1)
        # 这里的东西，相当于是要等模型完成最后一次推理，所以这里只需要判定模型是否完成了最新一轮输出或者文字长度是否已经发生了改变，判断这俩就够用了
        if(check_out_label or len(laststr) != laststr_len) :
            endSurcess = True
            break

    STATUS = LmActionType.STANDBY
    laststr=""
    return endSurcess

def get_status():
    global STATUS
    """
    获取当前运行状态
    return 当前的运行状态
    """
    return STATUS
def get_laststr():
    global laststr
    global cacheTimes
    """
    获取当前缓存字符
    return 当前的缓存字符,当前的token均值
    """
    cc= np.array(cacheTimes)
    return laststr, 1 / np.mean(cc),np.sum(cc)

def set_system_context(context):
    """
    设置system 的context
    params context : system的目标context
    """
    global llm_system_connect 
    llm_system_connect = context

def load_system_context():
    global llm_system_connect
    
    def_system = "You are alibaba, created by Alibaba Cloud. You are a helpful assistant."
    def_system = ""
    mcp_promot = asyncio.run(mcp_work.run_func(mcp_work.get_tools))
    tools =  build_tool_prompt(mcp_promot)

    final_system_content = f"{llm_system_connect if len(llm_system_connect) > 0 else def_system}\n{tools}"

    result = [{"role": "system", "content": final_system_content}]
    return result

def execute_tool(tool_name, **parameters):
    parameters['mcp_func']=tool_name
    rvalue = asyncio.run(mcp_work.run_func(mcp_work.call_tools,**parameters))
    return json.loads(rvalue)


def mcp_run(llm_response,conversation_history):
    max_turns = 5 # 限制最大对话轮次，防止无限循环
    current_turn = 0
    response_content = llm_response
    
    extracted_data = []
    while current_turn < max_turns:
        current_turn += 1
        print(f"\n--- Turn {current_turn} ---")
        

        """
        从文本中提取 <tool_code>...</tool_code> 内部的 JSON 内容。
        支持多行内容和多个工具代码块。
        """
        # 匹配 <tool_code> 和 </tool_code> 之间的内容，并将其捕获
        # re.DOTALL (re.S) 标志让 '.' 也能匹配换行符
        pattern = re.compile(r"<tool_code>(.*?)</tool_code>", re.DOTALL)
        
        find = False
        # 使用 finditer 查找所有非重叠匹配，并返回迭代器
        for match in pattern.finditer(response_content):
            find = True
            tool_call_str = match.group(1).strip() # match.group(1) 获取第一个捕获组的内容
            tool_call_data = json.loads(tool_call_str)
 
            try:
                # 解析工具调用指令
                tool_name = tool_call_data["tool_name"]
                parameters = tool_call_data.get("parameters", {})

                # 将大模型的工具调用输出添加到对话历史
                # conversation_history.append({"role": "assistant", "content": tool_call_str})

    
                # 5. 执行工具
                tool_result = execute_tool(tool_name, **parameters)
                
                # 6. 将工具结果作为上下文回传给大模型
                tool_result_message = {
                    "role": "tool", # 或者 'function' role in OpenAI's API
                    "name": tool_name,
                    "content": json.dumps(tool_result, ensure_ascii=False)
                }
                conversation_history.append(tool_result_message)
                
                # 再次循环，让大模型根据工具结果生成回答


                rendered_chat = llm_model.compiled_template.render(
                    messages=conversation_history, add_generation_prompt=True, **llm_model.template_kwargs
                )

                encoding = local_token.encode_batch(
                        [rendered_chat],
                        add_special_tokens=True,
                        is_pretokenized=False,
                    )
                    
                input_ids = encoding[0].ids
                input_ids = np.array(input_ids,np.int64)
                output = llm_model.generate(input_ids,None,local_token)
                response_content  = local_token.decode(output,skip_special_tokens=True)
                conversation_history.append({"role": "assistant", "content": response_content})

                continue # 进入下一轮循环，LLM会看到Tool Result并继续生成
                
            except json.JSONDecodeError:
                print("Error: LLM returned invalid tool_code JSON.")
                conversation_history.append({"role": "tool", "name": "error", "content": "LLM returned invalid tool_code JSON."})
                break # 终止循环
            except KeyError:
                print("Error: LLM returned malformed tool_code (missing name/parameters).")
                conversation_history.append({"role": "tool", "name": "error", "content": "LLM returned malformed tool_code."})
                break # 终止循环
        if(not find):
            break

    return conversation_history
        
        

def progress(ntoken,tokenizer,lstr):
    """
    进行解码，并与当前的laststr进行合并
    params ntoken : 模型的输出内容
    params tokenizer : 用于解码的token
    """
    global laststr
    global cache_ids 
    global STATUS
    global cacheNow
    STATUS = LmActionType.CONTINUE
    

    time123 = time.time()-cacheNow
    cacheTimes.append(time123)  
    cacheNow = time.time()
    cache_ids.append(ntoken[0][0])

    dstr =  tokenizer.decode(cache_ids,skip_special_tokens=True)
    if("�" in dstr):
        dstr = ""
    else:
        cache_ids=[]
        laststr += dstr


    return laststr



def thread_speak(context):
    """
    开始异步对话
    @params context: 输入的文本内容
    """
    global local_token
    global STATUS
    global laststr
    global message
    global llm_model
    global cacheNow
    global cacheTimes
    # if(len(message) == 0):
    laststr = ""


    input_message = load_system_context()
    

    # 读取历史对话记录
    for m in message:
        input_message.append(m)

  
    input_message.append( {"role": "user", "content": context})

    rendered_chat = llm_model.compiled_template.render(
        messages=input_message, add_generation_prompt=True, **llm_model.template_kwargs
    )

    encoding = local_token.encode_batch(
            [rendered_chat],
            add_special_tokens=True,
            is_pretokenized=False,
        )
        
    input_ids = encoding[0].ids
    input_ids = np.array(input_ids,np.int64)
    cacheNow = time.time()
    cacheTimes = []
    output = llm_model.generate(input_ids,progress,local_token)
    output  = local_token.decode(output,skip_special_tokens=True)


    input_message = mcp_run(output,input_message)

    STATUS = LmActionType.STOP
    # laststr = output

    # 将本轮模型的输出结果和用户的对话结果记录下来
    cc = np.array(cacheTimes)
    cc=np.nan_to_num(cc,0)
    mean_tokens = 1 / np.mean(cc)
    sum_tokens = np.sum(cc)

    if(np.isnan(mean_tokens)):
        mean_tokens = 0.0
    if(np.isnan(sum_tokens)):
        sum_tokens = 0.0

    message.append({"role": "user", "content": context})
    message.append({"role": "assistant", "content": laststr,"mean_tokens" :mean_tokens,"sum_tokens":sum_tokens})

def speak(context):
    """
    建立一个线程,开始异步对话
    @params context: 输入的文本内容
    return  msg:配置返回字符
    """
    global STATUS
    if(STATUS != LmActionType.STANDBY):
        STATUS = LmActionType.START
        # 创建一个线程，目标为上面定义的函数，并传递参数
        x = Thread(target=thread_speak, args=(context,))
        # 启动线程
        x.start()
        # 我们先不做等待，相当于这里不做长连接
        # x.join()
        return ""
    else:
        return f"error ! STATUS was {STATUS} "


def select_llm_model(modelkeys):
    """
    选中指定模型
    @params modelkeys: 选中的模型
    return  msg:配置返回字符
    """
    global llm_model
    global local_token

    if(modelkeys in llm_model_list.keys()):
        m_path = llm_model_list[modelkeys]['model_path']
        llm_model = llm_model_list[modelkeys]['model']
        llm_model = llm_model.QwenMoelRun(m_path)
        local_token = Tokenizer.from_file(m_path+"/tokenizer.json")
        return ""
    else:
        return "error cannot select this llm model"
    
def check_llm_model():
    """
    校队当前模型是否已加载
    return  msg:是否加载成功,True为加载成功,false为没有
    """
    global llm_model
    return False if llm_model is None else True

def history_message():
    global message
    return message

def read_yaml_config(file_path):
    """
    读取 YAML 配置文件，并确保以 UTF-8 编码处理。

    Args:
        file_path (str): YAML 文件的路径。

    Returns:
        dict: 解析后的 YAML 内容。
              如果文件不存在或解析失败，则返回 None。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到。")
        return None
    except yaml.YAMLError as exc:
        print(f"错误：解析 YAML 文件时出错: {exc}")
        return None


def build_tool_prompt(tools_data):
    """
    将工具数据构建成大模型可理解的提示词部分。
    """

    message = f"""你是一个智能助手，拥有解决复杂问题的能力。请按照以下步骤来完成你的任务：
    1.  **根据用户请求，决定是否需要使用外部工具。**
    2.  **如果需要，请严格按照以下格式输出工具调用指令。在输出工具指令前，请先思考你为什么要调用此工具。**
    3.  **在工具执行完成后，你会收到工具的输出。请根据工具输出的内容，继续完成任务或直接回答用户。**

    

    1. **思考 (Thought):** 仔细分析用户请求，判断需要哪些信息，以及可以利用哪些工具来获取这些信息。
    2. **行动 (Action):** 如果需要调用工具，请按照以下格式输出：
    CALL: <工具名称>(<参数1>=<值1>, <参数2>=<值2>, ...)
    确保所有必填参数都已提供。
    3. **观察 (Observation):** 每次工具调用后，我会将工具的执行结果反馈给你。你需要根据观察结果，继续你的思考和行动，直到问题得到解决。
    4. **最终回答:** 当你拥有足够的信息可以回答用户问题时，直接给出最终答案。
    **可用工具：**
    """

  
    mcp_message = r'这里的 "MCP" 指的是 **多能力平台 (Multi-Capability Platform)**，它集成了一系列外部工具和接口，例如查询天气、获取位置信息等。当你看到 "MCP" 时，请理解它代表我们可调用的外部工具能力集合。'

    tools_data = json.loads(tools_data)
    if not tools_data or not tools_data.get('result'):
        return ""

    tool_descriptions = []
    for tool in tools_data['result']:
        name = tool['name']
        description = tool['description']
        parameters = tool.get('parameters', {})
        required_params = tool.get('required_params', [])

        param_strs = []
        for param_name, param_info in parameters.items():
            param_type = param_info.get('type', 'string')
            param_desc = param_info.get('description', '')
            param_default = param_info.get('default')
            
            param_str = f"- {param_name} ({param_type}): {param_desc}"
            if param_default is not None:
                param_str += f" (默认值: {param_default})"
            if param_name in required_params:
                param_str += " (必填)"
            param_strs.append(param_str)
        
        params_section = "\n  ".join(param_strs) if param_strs else "无"

        tool_descriptions.append(
            f"name: {name}\n"
            f"description: {description}\n"
            f"parameters:\n  {params_section}\n"
        )

    message_title = f"""
        你是一个功能强大的AI助手，能够理解并执行复杂任务。
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
        """
    config = read_yaml_config("./mcp/utils/mcp_config.yaml")
    message_title = config['mcp_title']
    message = f"""
        **可用工具：**
        {json.dumps(tool_descriptions, indent=2, ensure_ascii=False)}
        ----
        """
    
    

    return "\n\n" +message_title+ "\n---\n\n" + message