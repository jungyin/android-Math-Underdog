from infer.qwen.base_infer import BaseMoelRun 
DEFAULT_SYSTEM_PROMPT='''你是一名智能评估助手，专门判断检索到的文档是否能够有效回答用户的问题。'''
DEFAULT_REWRITE_PROMPT='''
请基于以下用户的搜索问题和检索到的文档，判断该文档是否能够有效回答用户的问题。  

**判断标准：**  
1. 如果文档中包含直接或间接回答用户问题的内容，返回 `1`。  
2. 如果文档内容与用户问题无关，或无法提供有效答案，返回 `0`。  
3. 仅返回 `0` 或 `1`，不包含其他解释或信息。  

**用户搜索问题：**  
"{query}"  

**检索到的文档内容：**  
"{document}"  

**判断结果：**
'''
from typing import List
import numpy as np


class LLMJudger():
    def __init__(self,tokenizer,llm_model:BaseMoelRun=None,system_prompt=DEFAULT_SYSTEM_PROMPT,rewrite_prompt=DEFAULT_REWRITE_PROMPT):
        self.llm_model = llm_model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.rewrite_prompt = rewrite_prompt

    def judge(self, query: str, documents: List[str]) -> List[int]:


        results = []
        for document in documents:
            prompt = self.rewrite_prompt.format(query=query, document=document)

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]

            rendered_chat = self.llm_model.compiled_template.render(messages=messages, add_generation_prompt=True, **self.llm_model.template_kwargs)

            

            model_inputs = self.tokenizer.encode_batch(
                [rendered_chat],
                add_special_tokens=True,
                is_pretokenized=False,
            )[0]

            input_ids = model_inputs.ids
            input_ids = np.array(input_ids,np.int64)

            output = self.llm_model.generate(input_ids,None,None)
            response = self.tokenizer.decode(output,skip_special_tokens=True)
            # 确保返回值是 0 或 1
            try:
                result = int(response.strip())
                if result not in [0, 1]:
                    result = 0  # 默认返回 0，防止异常值
            except ValueError:
                result = 0  # 如果解析失败，默认返回 0

            results.append(result)

        return results


if __name__ == '__main__':

    from infer.qwen.source_infer import QwenMoelRun
    from tokenizers import Tokenizer

    mpath = "D:/code/transformer_models/models--Qwen--Qwen2.5-0.5B-Instruct/"
    model = QwenMoelRun(mpath)
    tokenizer = Tokenizer.from_file(mpath+"tokenizer.json")

    llm_rewriter = LLMJudger(tokenizer,model)
    response=llm_rewriter.judge(query="请总结伊朗总统罹难事件",documents=["坠机事故发生后，伊朗最高领袖哈梅内伊表示，希望莱希及随行官员平安回家，同时国家和政府工作不会受影响。莱希等人确定罹难后，伊朗政府内阁举行了特别会议，将会宣布莱希等人的葬礼安排。伊朗政府内阁发布声明，“向国家和人民保证，将继续莱希总统的前进道路，国家的治理不会受到干扰”。根据伊朗宪法第131条，总统死亡、被免职、辞职、缺席或患病时间超过两个月，或总统任期结束由于某些原因而未选出新总统时，伊朗第一副总统应在最高领袖的批准下承担总统的权力和职责。由伊朗伊斯兰议会议长、司法总监和第一副总统组成的委员会有义务在最多五十天的时间内安排选举新总统。"])
    print(response)
    response=llm_rewriter.judge(query="请总结伊朗总统罹难事件",documents=["这个是一个无效信息"])
    print(response)
