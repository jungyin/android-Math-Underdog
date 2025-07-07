

from infer.qwen.base_infer import BaseMoelRun 
import numpy as np

DEFAULT_SYSTEM_PROMPT='''你是一名搜索优化专家，擅长改写用户查询，使其更适合搜索引擎处理。'''
DEFAULT_REWRITE_PROMPT='''
请基于以下用户输入的问题，扩展出多个相关的搜索词组，确保：  
1. 只扩展关键词，不改写为完整的问题句。  
2. 生成 3-5 组相关的搜索关键词，覆盖不同的搜索角度。  
3. 结合同义词、近义词、行业术语、相关概念，确保搜索范围更广。  
4. 结果用中文分号（`;`）分隔，不包含多余的解释或符号。  
5.关键词之间的词语可以用空格隔开，显示有层次感
**用户查询：**  
"{query}"  

**扩展后的关键词组：**
'''
class LLMRewriter():
    def __init__(self,tokenizer,llm_model:BaseMoelRun=None,system_prompt=DEFAULT_SYSTEM_PROMPT,rewrite_prompt=DEFAULT_REWRITE_PROMPT):
        self.llm_model = llm_model
        self.system_prompt = system_prompt
        self.rewrite_prompt = rewrite_prompt
        self.tokenizer = tokenizer
    def rewrite(self, query):

        prompt = self.rewrite_prompt.format(query=query)
        # print(prompt)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content":prompt }
        ]
        rendered_chat = self.llm_model.compiled_template.render(messages=messages, add_generation_prompt=True, **self.llm_model.template_kwargs)
   
        model_inputs = self.tokenizer.encode_batch(
            [rendered_chat],
            add_special_tokens=True,
            is_pretokenized=False,
        )[0]

        input_ids = model_inputs.ids
        input_ids = np.array(input_ids,np.int64)
      
        output = model.generate(input_ids,None,None)
        response = tokenizer.decode(output,skip_special_tokens=True)
        
        return response


if __name__ == '__main__':
    
    from infer.qwen.source_infer import QwenMoelRun
    from tokenizers import Tokenizer

    # mpath = "D:/code/transformer_models/models--Qwen--Qwen2.5-Coder-3B-Instruct-GPTQ-Int8/"
    mpath = "D:/code/transformer_models/models--Qwen--Qwen2.5-Coder-0.5B-Instruct-GPTQ-Int8/"

    tokenizer = Tokenizer.from_file(mpath+"tokenizer.json")
    model = QwenMoelRun(mpath)
    llm_rewriter = LLMRewriter(tokenizer,model)
    # response=llm_rewriter.rewrite(query="如何提高英语口语？")
    response=llm_rewriter.rewrite(query="请总结伊朗总统罹难事件")
    print(response)