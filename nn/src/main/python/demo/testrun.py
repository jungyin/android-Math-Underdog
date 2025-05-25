
from transformer_lite.generation.logits_process import LogitsProcessorList,RepetitionPenaltyLogitsProcessor,TopKLogitsWarper,TopPLogitsWarper,TemperatureLogitsWarper 
from transformer_lite.generation.stopping_criteria import StoppingCriteriaList,MaxLengthCriteria ,EosTokenCriteria
import numpy as np
import onnxruntime as ort

import torch

class QwenMoelRun():
    def __init__(self):
        # 模型参数的加载
        self.pad_token_id = 151643
        self.bos_token_id = 151643
        self.eos_token_id = [151645,151643]
        self.max_position_embeddings=32768
        # 是否存在最大长度
        self.has_default_max_length = True
        # 是否存在最短长度
        self.has_default_min_length = True
        # 最长tokean数
        self.max_new_tokens=512
        # 由于没有限制最小长度，这里配为0
        self. min_length = 0
         # 模型所在路径
        model = "qwen2-code-0.5b"
        model_type="onnx"
        self.model_path ="./" + model + "/" + model_type + "/" + "model.onnx"
        self.model_path ="./" + model + "/" + model_type + "/" + "model32.onnx"
        self.lm_model_path ="./" + model + "/" + model_type + "/" + "lm_model32.onnx"
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        # session_options.enable_cuda_graph = True  # 如果需要启用 CUDA 图形优化
        # session_options.gpu_id = 0  # 指定使用第 0 块 GPU
        # 有一个linear层无法导出来，这里需要手动额外加载使用
        self.lm_head = ort.InferenceSession(self.lm_model_path, sess_options=session_options, providers=['CUDAExecutionProvider'])
        # 模型本体的加载
        print("Model is valid and supported by the current ONNX Runtime.")
        self.model = ort.InferenceSession(self.model_path, sess_options=session_options, providers=['CUDAExecutionProvider'])


# 预处理 qwen2的输入数据
    def prepare_inputs_for_generation(self, input_ids,position_ids=None, attention_mask=None, past_key_values=None,  inputs_embeds=None,seen_tokens=0, **kwargs):

        if attention_mask is None:
            attention_mask = np.ones_like(input_ids,dtype=np.int64)
        past_length = seen_tokens
        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]

            
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation

            position_ids = np.cumsum(attention_mask,axis=-1).astype(np.int64)-1
            position_ids = np.where(attention_mask==0,1,position_ids)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs



    def softmax(self,x,axis=-1):


        # 计算输入向量的最大值，并保持其原始维度
        x_max = np.max(x, axis=axis, keepdims=True)
        # 对平移后的向量进行指数运算
        e_x = np.exp(x - x_max)
        # 计算指数运算结果的和，并保持其原始维度
        return e_x / np.sum(e_x, axis=axis, keepdims=True)
    def multinomial_numpy(self,probs, num_samples=1):
        # 获取批次大小和词汇表大小
        batch_size, vocab_size = probs.shape
        # 初始化结果数组
        next_tokens = np.zeros(batch_size, dtype=np.int64)
        # 对每个样本单独采样
        for i in range(batch_size):
            # 使用 numpy.random.choice 按照概率分布采样
            next_tokens[i] = np.random.choice(vocab_size, size=num_samples, p=probs[i], replace=True)
        return next_tokens.squeeze()  # 如果 num_samples=1，则去掉多余的维度
    
    
    # 这里模拟ForCausalLM方法
    def runForCausalLM(self ,input_ids):

        input_names = [input.name for input in self.model.get_inputs()]
        inputs = self.prepare_inputs_for_generation(input_ids,None,None,None)
        # inputs = {"input_ids": inputs['input_ids'],"attention_mask":inputs["attention_mask"],"position_ids":inputs["position_ids"]}
        output_names = [output.name for output in self.model.get_outputs()]
        outputs=self.model.run(output_names,inputs)
        hidden_states = outputs[0]

        llm_inputs = {"input_0": hidden_states}
        llm_output_names = [output.name for output in self.lm_head.get_outputs()]
        logits = self.lm_head.run(llm_output_names,llm_inputs)
        logits = logits[0].astype(np.float32)
        return logits
    def generate(self,input_ids,stream=None,tokenizer=None):
       
        input_ids = np.array(input_ids)
        input_ids = np.reshape(input_ids,[1,input_ids.shape[0]])
        # 获取当前输入内容的长度
        input_ids_length = input_ids.shape[1]
        # 重新编译最大长度。
        max_len = input_ids_length + self.max_new_tokens
  
        # 校队有没有超长
        if input_ids_length >= max_len:
            input_ids_string = "input_ids"
            raise ValueError(
                f"Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to"
                f" {max_len}. This can lead to unexpected behavior. You should consider"
                " increasing `max_length` or, better yet, setting `max_new_tokens`."
            )


         # 用logits processor 来控制生成的过程
        self.logits_processor =  LogitsProcessorList()
        # 用stopping_criteria控制结束的过程
        self.stopping_criteria = StoppingCriteriaList()
        # 调整生成文本的随机性
        self.logits_warper = LogitsProcessorList()

        temperature = 0.7
        top_k = 20
        top_p = 0.8
        min_tokens_to_keep=1

        # 添加 RepetitionPenaltyLogitsProcessor
        repetition_penalty = 1.05
        # 追加生成过程器
        self.logits_processor.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        # 追加停止器
        self.stopping_criteria.append(MaxLengthCriteria(max_length=max_len,max_position_embeddings=self.max_position_embeddings))
        self.stopping_criteria.append(EosTokenCriteria(self.eos_token_id))

        # 追加??
        self.logits_warper.append(TemperatureLogitsWarper(temperature))
        self.logits_warper.append(TopKLogitsWarper(top_k=top_k,min_tokens_to_keep=min_tokens_to_keep))
        self.logits_warper.append(TopPLogitsWarper(top_p=top_p,min_tokens_to_keep=min_tokens_to_keep))
        


        expand_size=1
        input_ids = np.repeat(input_ids,expand_size, 0)
        

        batch_size, seq_length = input_ids.shape

        unfinished_sequences = np.ones(batch_size, dtype=np.int64)
        unfinished_sequences = torch.from_numpy(unfinished_sequences)


        

        this_peer_finished = False
        # 这里是尝试模拟sample中，通过_has_unfinished_sequences方法来while循环执行的过程
        
        scores = None

        first = True
        lstr = ''
        while(not this_peer_finished):
    
            logits = self.runForCausalLM(input_ids)
            input_ids = torch.from_numpy(input_ids)
            logits = torch.from_numpy(logits)


            next_token_logits = logits[:,-1,:]
            next_token_scores = self.logits_processor(input_ids, next_token_logits)
            next_token_scores = self.logits_warper(input_ids, next_token_scores)
            next_token_scores = next_token_scores.numpy()
            probs = self.softmax(next_token_scores,-1)
            next_tokens = self.multinomial_numpy(probs,1)
            next_tokens = torch.from_numpy(next_tokens)
            if self.eos_token_id is not None:
                next_tokens = next_tokens * unfinished_sequences + self.pad_token_id * (1 - unfinished_sequences)
            # 更新input_ids,将inputid与新的输出内容进行拼接
            ntoken = next_tokens[:, None]
            input_ids = np.concatenate([input_ids, ntoken], axis=-1)

            if(stream is not None):
                lstr = stream(ntoken,tokenizer,lstr)

            input_ids = torch.from_numpy(input_ids)
            if not (scores is None):
                scores = torch.from_numpy(scores)
            unfinished_sequences = unfinished_sequences & ~self.stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            input_ids = input_ids.numpy()
        #这里结束推理，进行下一步操作
        return input_ids[0][input_ids_length:]
if __name__ == "__main__":
    qwen = QwenMoelRun()
    