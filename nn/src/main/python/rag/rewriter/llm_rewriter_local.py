
from infer.qwen.source_infer import QwenMoelRun

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
    def __init__(self,llm_model:BaseModel=None,system_prompt=DEFAULT_SYSTEM_PROMPT,rewrite_prompt=DEFAULT_REWRITE_PROMPT):
        self.llm_model = llm_model
        self.system_prompt = system_prompt
        self.rewrite_prompt = rewrite_prompt
    def rewrite(self, query):

        prompt = self.rewrite_prompt.format(query=query)
        # print(prompt)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        text = self.llm_model.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print(text)
        model_inputs = self.llm_model.tokenizer([text], return_tensors="pt").to(self.llm_model.device)

        generated_ids = self.llm_model.model.generate(
            model_inputs.input_ids,
            max_new_tokens=1024,
            do_sample=False,
            top_k=10
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.llm_model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


if __name__ == '__main__':

    model = QwenChat("D:/code/transformer_models/models--Qwen--Qwen2.5-Coder-3B-Instruct-GPTQ-Int8")
    llm_rewriter = LLMRewriter(model)
    # response=llm_rewriter.rewrite(query="如何提高英语口语？")
    response=llm_rewriter.rewrite(query="请总结伊朗总统罹难事件")
    print(response)