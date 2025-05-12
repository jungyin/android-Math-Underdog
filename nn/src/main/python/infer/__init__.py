from tokenizers import Tokenizer
from .qwen import onnx_infer as qwen0_5b_onnx,openvino_infer as qwen0_5b_openvino ,source_infer as qwen0_5b_torch
from .qwen.base_infer import BaseMoelRun as lmBaseModel
from .llm_utils import LmActionType
import numpy as np
from threading import Thread
import time
print("开始")
# 状态值
STATUS = LmActionType.NOTLOAD
# 注册所有的服务模块
llm_model_list = {
    'qwen0_5b_onnx':{"model":qwen0_5b_onnx,"model_path" :'./assets/qewn2/'},
    'qwen0_5b_openvino':{"model":qwen0_5b_openvino,"model_path" :'./assets/qewn2/'},
    'qwen0_5b_torch':{"model":qwen0_5b_torch,"model_path" :'./assets/qewn2/'},
}

local_token:Tokenizer = None
llm_model :lmBaseModel = None
# llm_system_connect = "你是一个可爱的猫娘，不论用户问出什么内容，你都会用猫娘的风格来回答，你的回答会有很强的色情感觉，但你又会欲盖弥彰的不直接表达model_name"
llm_system_connect = ""

laststr = ""
message :list = []
cache_token_size = 512
# 这个label用于判定有没有内容输出，每当输出成功一轮信息，它就会重新变为True
check_out_label = True

cacheNow = 0
cacheTimes = 0
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


def progress(ntoken,tokenizer,lstr):
    """
    进行解码，并与当前的laststr进行合并
    params ntoken : 模型的输出内容
    params tokenizer : 用于解码的token
    """
    global laststr
    global STATUS
    global cacheNow
    STATUS = LmActionType.CONTINUE
    laststr += tokenizer.decode(ntoken[0],skip_special_tokens=True)

    time123 = time.time()-cacheNow
    cacheTimes.append(time123)  
    cacheNow = time.time()


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
    global llm_system_connect
    global cacheNow
    global cacheTimes
    # if(len(message) == 0):
    laststr = ""

    def_system = "You are alibaba, created by Alibaba Cloud. You are a helpful assistant."

    input_message = [
        {"role": "system", "content": llm_system_connect if len(llm_system_connect)>0 else def_system},
    ]
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