
from flask import  jsonify      
import infer
import numpy as np
infer.select_llm_model("qwen0_5b_openvino")

def llm_select(req):
    """
    选定指定的模型，如果选定接口报错500，选定成功正常
    return : m_list :模型列表,m_num:模型数量
    """
    modelkey = req.args["model"]
    if(modelkey in infer.llm_model_list.keys):
        infer.select_llm_model(modelkey)
        return jsonify({"msg":f"surcess select this {modelkey} model!"}),200
    else:
        return jsonify({"msg":f"eror!cannot check this model:{modelkey}"}),500

def set_system_context(req):
    """
    配置system信息
    params  system_context : system的配置信息
    return  msg:配置返回字符
    """
    system_context = req.args['system_context']
    infer.set_system_context(system_context)
    return jsonify({"msg":f"surcess write!"}),200
    
def restart_all(req):
    """
    重置对话
    return  msg:配置返回字符
    """
    infer.restart_all()
    return jsonify({"msg":f"surcess work!"}),200
    
def get_llm_out(req):
    """
    获取最新的对话内容，完整的string内容
    return  data:对话内容{context: 对话内容，status}
    """

    lstr,mean_token,sum_tokens=infer.get_laststr()
    mean_token=np.nan_to_num(mean_token,0)
    sum_tokens=np.nan_to_num(sum_tokens,0)

    return jsonify({"msg":f"surcess read!",
                    "data":{"context":lstr,"mean_tokens":mean_token,"sum_tokens":sum_tokens,"status":infer.get_status()}}),200

def get_llm_history(req):
    """
    获取最新的对话内容，完整的string内容
    return  data:历史对话内容
    """
    return jsonify({"msg":f"surcess read!",
                "data":infer.history_message()}),200


def stop_speak(req):
    """
    停止一轮对话
    return  msg:配置返回字符
    """
    es = infer.stop_speak()
    return jsonify({"msg":f"surcess end!","data":{"work": 1 if es else 0,"status":infer.STATUS}}),200

def speek(req):
    """
    进行一次对话
    params  context : 对话内容
    return  msg:配置返回字符
    """
    if(infer.check_llm_model()):
        context = req.args['context']
        if context is None or context == '':
            return jsonify({"msg":f"eror! Illegal parameter from context"}),500
        rstr = infer.speak(context)
        if len(rstr) == 0:
            return jsonify({"msg":f"surcess! start speak!"}),200
        else :
            return jsonify({"msg":f"error! message:{rstr} "}),500
    else:
        return jsonify({"msg":f"eror! check model faile"}),500
    
def llm_models(req):
    """
    获取可用模型的列表
    return : m_list :模型列表,m_num:模型数量
    """
    modellist = infer.llm_model_list.keys

    return jsonify({"data":{"m_list":modellist,"m_num":len(modellist)}}),200