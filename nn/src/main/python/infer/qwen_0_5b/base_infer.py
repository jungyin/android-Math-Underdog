
from np_logits.logits_process import RepetitionPenaltyLogitsProcessor,TemperatureLogitsWarper,TopKLogitsWarper,TopPLogitsWarper , MaxLengthCriteria ,EosTokenCriteria
import numpy as np
from nputils import softmax,multinomial_numpy


class BaseMoelRun():
    def __init__(self):
        # 模型参数的加载
        self.pad_token_id = 151643
        self.bos_token_id = 151643
        self.eos_token_id = [151645,151643]
        self.max_position_embeddings=32768
        self.max_len = 0
        # 是否存在最大长度
        self.has_default_max_length = True
        # 是否存在最短长度
        self.has_default_min_length = True
        # 最长tokean数
        self.max_new_tokens=512
        # 由于没有限制最小长度，这里配为0
        self. min_length = 0
        

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

    def logits_warper(self,inputids,next_token_scores):
    
        temperature = 0.7
        top_k = 20
        top_p = 0.8
        min_tokens_to_keep=1

        tlw = TemperatureLogitsWarper(temperature)
        tk = TopKLogitsWarper(top_k=top_k,min_tokens_to_keep=min_tokens_to_keep)
        tp = TopPLogitsWarper(top_p=top_p,min_tokens_to_keep=min_tokens_to_keep)

        score = tlw(inputids, next_token_scores)
        score = tk(inputids, next_token_scores)
        score = tp(inputids, next_token_scores)
        return score
    def stopping_criteria(self,inputids,next_token_scores):
        criterias = []
       
        criterias.append(MaxLengthCriteria(max_length=self.max_len,max_position_embeddings=self.max_position_embeddings))
        criterias.append(EosTokenCriteria(self.eos_token_id))
        is_done = False

        for criteria in  criterias:
            is_done = is_done | criteria(inputids, next_token_scores)
        return is_done
    
    # 这里模拟ForCausalLM方法
    def runForCausalLM(self ,input_ids):
        pass
    def generate(self,input_ids,stream=None,tokenizer = None):
       
        input_ids = np.array(input_ids)
        input_ids = np.reshape(input_ids,[1,input_ids.shape[0]])
        # 获取当前输入内容的长度
        input_ids_length = input_ids.shape[1]
        # 重新编译最大长度。
        self.max_len = input_ids_length + self.max_new_tokens
  
        # 校队有没有超长
        if input_ids_length >= self.max_len:
            input_ids_string = "input_ids"
            raise ValueError(
                f"Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to"
                f" {self.max_len}. This can lead to unexpected behavior. You should consider"
                " increasing `max_length` or, better yet, setting `max_new_tokens`."
            )

   
        # 添加 RepetitionPenaltyLogitsProcessor
        repetition_penalty = 1.05
        # 用logits processor 来控制生成的过程
        self.logits_processor = RepetitionPenaltyLogitsProcessor(repetition_penalty)
        # 用stopping_criteria控制结束的过程

        


        expand_size=1
        input_ids = np.repeat(input_ids,expand_size, 0)
        

        batch_size, seq_length = input_ids.shape

        unfinished_sequences = np.ones(batch_size, dtype=np.int64)

        this_peer_finished = False
        # 这里是尝试模拟sample中，通过_has_unfinished_sequences方法来while循环执行的过程
        
        scores = None

        first = True
        lstr=''
        while(not this_peer_finished):
    
            logits = self.runForCausalLM(input_ids)


            next_token_logits = logits[:,-1,:]
            next_token_scores = self.logits_processor(input_ids, next_token_logits)
            next_token_scores = self.logits_warper(input_ids, next_token_scores)
            next_token_scores = next_token_scores
            probs = softmax(next_token_scores,-1)
            next_tokens = multinomial_numpy(probs,1)
            if self.eos_token_id is not None:
                next_tokens = next_tokens * unfinished_sequences + self.pad_token_id * (1 - unfinished_sequences)
            # 更新input_ids,将inputid与新的输出内容进行拼接
            ntoken = next_tokens[:, None]
            input_ids = np.concatenate([input_ids, ntoken], axis=-1)
            if(stream is not None):
                lstr = stream(ntoken,tokenizer,lstr)

            endv = not self.stopping_criteria(input_ids, scores)
            unfinished_sequences = unfinished_sequences &  endv
            this_peer_finished = unfinished_sequences.max() == 0
            input_ids = input_ids
        #这里结束推理，进行下一步操作
        return input_ids[0][input_ids_length:]