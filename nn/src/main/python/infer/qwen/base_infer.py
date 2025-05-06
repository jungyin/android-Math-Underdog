
from np_logits.logits_process import RepetitionPenaltyLogitsProcessor,TemperatureLogitsWarper,TopKLogitsWarper,TopPLogitsWarper , MaxLengthCriteria ,EosTokenCriteria
import numpy as np
from nputils import softmax,multinomial_numpy
from ..base_transformer import BaseMoel
import torch
class BaseMoelRun(BaseMoel):
    def __init__(self,model_assets):
        super().__init__(model_assets)
        self.runmode = "sample"
        # self.runmode = "greedy_search"
        past_key_value_type = np.float32
        self.empty_past_key_value = self._build_past_key_value(past_key_value_type)

    def _build_past_key_value(self,dtype:np.dtype = np.float16):
        
        num_hidden_layers = self.config['num_hidden_layers']
        head_dim = self.config['hidden_size'] // self.config['num_attention_heads']
        num_kvheads = self.config['num_key_value_heads']
        kvszie = 2 
        # 1,2,?,head_dim
        
        qvshape = [1,num_kvheads,0,head_dim]
        
        return np.zeros([num_hidden_layers,kvszie]+qvshape,dtype=dtype)

# 预处理 qwen2的输入数据
    def prepare_inputs_for_generation(self, input_ids,position_ids=None, attention_mask=None, past_key_values=None,  inputs_embeds=None):

        if attention_mask is None:
            attention_mask = np.ones_like(input_ids,dtype=np.int64)
        

        if(past_key_values is None):
            past_length = 0
        else:
            past_length = past_key_values[0][0].shape[2]

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
            if past_key_values is not None:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        if past_key_values is None:
            past_key_values = self.empty_past_key_value


        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def logits_warper(self,inputids,next_token_scores):
    
 
        tlw = TemperatureLogitsWarper(self.temperature)
        tk = TopKLogitsWarper(top_k=self.top_k,min_tokens_to_keep=self.min_tokens_to_keep)
        tp = TopPLogitsWarper(top_p=self.top_p,min_tokens_to_keep=self.min_tokens_to_keep)

        score = tlw(inputids, next_token_scores)
        score = tk(inputids, next_token_scores)
        score = tp(inputids, next_token_scores)
        return score
    
    # 停止方法，这里的话，qwen暂时只有sample和
    def stopping_criteria(self,inputids,next_token_scores):
        criterias = []
       
        criterias.append(MaxLengthCriteria(max_length=self.max_len,max_position_embeddings=self.max_position_embeddings))
        criterias.append(EosTokenCriteria(self.eos_token_id))
        is_done = False

        for criteria in  criterias:
            is_done = is_done | criteria(inputids, next_token_scores)
        return is_done
    
    # 这里模拟ForCausalLM方法
    def runForCausalLM(self ,input_ids,past_key_values=None):
        pass

    # 停止本轮生成
    def stopGenerate(self):
        self.run_gen = False

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

        lstr=''
        past_key_values = None
        self.run_gen = True
        while(not this_peer_finished) and self.run_gen:    
            outputs = self.runForCausalLM(input_ids,past_key_values)
            logits = outputs[0]
            past_key_values = outputs[1]

            if(logits is np.array):
                logits = logits.astype(np.float32)
            elif logits is torch.Tensor:
                logits =logits.to(torch.float32)

            next_token_logits = logits[:,-1,:]
            next_token_scores = self.logits_processor(input_ids, next_token_logits)
            if(self.runmode == "sample"):
                next_token_scores = self.logits_warper(input_ids, next_token_scores)
            next_token_scores = next_token_scores
          


            # sample 和 greedy_search 的计算方法有一点不同，这里给区分开了
            if(self.runmode == "sample"):
                probs = softmax(next_token_scores,-1)
                next_tokens = multinomial_numpy(probs,1)
                if self.eos_token_id is not None:
                    next_tokens = next_tokens * unfinished_sequences + self.pad_token_id * (1 - unfinished_sequences)
            elif(self.runmode == "greedy_search"):
                next_tokens = np.argmax(next_token_scores,axis=-1)


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
    