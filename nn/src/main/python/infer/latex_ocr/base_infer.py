
from np_logits.logits_process import RepetitionPenaltyLogitsProcessor,TemperatureLogitsWarper,TopKLogitsWarper,TopPLogitsWarper , MaxLengthCriteria ,EosTokenCriteria
import numpy as np
from nputils import softmax,multinomial_numpy
from ..base_transformer import BaseMoel
import cv2

import time
class BaseMoelRun(BaseMoel):
    def __init__(self,model_assets):
        super().__init__(model_assets)

# 预处理 qwen2的输入数据
    def prepare_inputs_for_generation(self, input_ids,position_ids=None, attention_mask=None, past_key_values=None, **kwargs):

        if attention_mask is None:
            attention_mask = np.ones_like(input_ids,dtype=np.int64)
       
   
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]


                # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]



        model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def logits_warper(self,inputids,next_token_scores):
    

        return next_token_scores
    def stopping_criteria(self,inputids,next_token_scores):
        criterias = []
       
        criterias.append(MaxLengthCriteria(max_length=self.config['max_length'],max_position_embeddings=self.max_position_embeddings))
        criterias.append(EosTokenCriteria(self.eos_token_id))
        is_done = False

        for criteria in  criterias:
            is_done = is_done | criteria(inputids, next_token_scores)
        return is_done
    
    def greedy_search(self,frame,stream=None,tokenizer = None):
        
        input_frame = self.build_frame(frame)

        ctime = time.time()
        encode = self.encoder(input_frame)
        print("检查1：",time.time()-ctime)

        input_ids = np.array([[0]])
        batch_size, seq_length = input_ids.shape

        unfinished_sequences = np.ones(batch_size, dtype=np.int64)

        # 这里是尝试模拟sample中，通过_has_unfinished_sequences方法来while循环执行的过程
        this_peer_finished = False
        scores = None

        first = True
        lstr=''
        past_key_values = None


        while(not this_peer_finished):
            model_inputs = self.prepare_inputs_for_generation(input_ids,None,past_key_values)

            logits = self.decoder(model_inputs['input_ids'].astype(np.int64),model_inputs['attention_mask'],np.expand_dims(encode,0))

            
            next_token_logits = logits[:,-1,:]
            next_tokens = np.argmax(next_token_logits,axis=-1)

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
            
        print("检查2：",time.time()-ctime)
        #这里结束推理，进行下一步操作
        return input_ids[0]
    
    def encoder(self,pixel_values):
        pass

    def decoder(self,input_ids,attention_mask,encoder_hidden_states):
        pass

    def build_frame(self,frame):
        image_mean = 0.5
        image_std = 0.5
        rescale_factor = 0.00392156862745098
        size = [500,400]

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame,size)
        frame = frame * rescale_factor
        frame = np.transpose(frame,(2,0,1))
        frame = (frame - image_mean) / image_std 

        frame = frame.astype(np.float32)

        return np.expand_dims(frame,0)
