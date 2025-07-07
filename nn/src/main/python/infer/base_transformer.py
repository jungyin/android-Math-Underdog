
from np_logits.logits_process import RepetitionPenaltyLogitsProcessor,TemperatureLogitsWarper,TopKLogitsWarper,TopPLogitsWarper , MaxLengthCriteria ,EosTokenCriteria
import numpy as np
from nputils import softmax,multinomial_numpy
import os
import json
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment
class BaseMoel():
    def __init__(self,model_assets):

        if not str(model_assets).endswith("/"):
            model_assets + model_assets+"/"
            
        config = self._dict_from_json_file(model_assets+"config.json")
        token_config = self._dict_from_json_file(model_assets+"tokenizer_config.json")
        self.token_config = token_config
        self.config = config
        generation_config = self._dict_from_json_file(model_assets+"generation_config.json")
        self.generation_config = generation_config
        
        # 模型参数的加载
        self.pad_token_id = generation_config['pad_token_id']
        self.bos_token_id = generation_config['bos_token_id']
        self.eos_token_id = generation_config['eos_token_id']
        if("max_length" in config):
            self.max_position_embeddings=config['max_length']
        else:
            self.max_position_embeddings=None
        self.max_len = 0
        self.min_length = 0
        # 是否存在最大长度
        self.has_default_max_length = self.max_position_embeddings is not None
        # 是否存在最短长度
        self.has_default_min_length = self.min_length is not None

        # 最长tokean数
        self.max_new_tokens=1024
        # 由于没有限制最小长度，这里配为0
        self. min_length = 0
        if ("temperature" in generation_config):
            self.temperature = generation_config['temperature']

        if ("top_k" in generation_config):
            self.top_k = generation_config['top_k']
        if ("top_p" in generation_config):
            self.top_p = generation_config['top_p']
        self.min_tokens_to_keep=1
        self.model_f = "./model_file/"


        def raise_exception(message):
            raise TemplateError(message)
        jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        jinja_env.globals["raise_exception"] = raise_exception
        if('chat_template' in token_config):
            chat_template = token_config['chat_template']
            self.compiled_template = jinja_env.from_string(chat_template)
        else:
            self.compiled_template = None
        self.template_kwargs = {
            'eos_token' : token_config['eos_token'],
            'pad_token' : token_config['pad_token'],
        }
        if("additional_special_tokens" in token_config):
           self.template_kwargs["additional_special_tokens"]  = token_config['additional_special_tokens']
       
    def _dict_from_json_file(self, json_file):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)
    

