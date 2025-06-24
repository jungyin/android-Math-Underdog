
from np_logits.logits_process import RepetitionPenaltyLogitsProcessor,TemperatureLogitsWarper,TopKLogitsWarper,TopPLogitsWarper , MaxLengthCriteria ,EosTokenCriteria
import numpy as np
from nputils import softmax,multinomial_numpy
from ..base_transformer import BaseMoel
import cv2
import re
import time
class BaseMoelRun(BaseMoel):
    def __init__(self,model_assets):
        super().__init__(model_assets)

    def process_mixed_content(self,full_string):
        """
        处理包含特殊标签和数学公式的混合字符串。

        Args:
            full_string (str): 包含 <s><s>, </s> 和 \[ \] 数学公式的字符串。

        Returns:
            list: 一个列表，每个元素是一个字典，表示文本块或公式块，
                例如：[{"type": "text", "content": "..."}] 或
                [{"type": "formula", "content": "..."}]
        """
        # 1. 移除最外层的 <s><s> 和 </s> 标签
        cleaned_string = full_string.replace("<s><s>", "").replace("</s>", "").strip()

        # 2. 定义正则表达式来匹配数学公式块（包括其前后的空白符），并捕获公式内容
        # re.DOTALL 确保 . 匹配包括换行符在内的所有字符
        # 外层捕获组 () 是为了让 re.split 保留分隔符本身
        math_formula_pattern = r'(\s*\\\[.*?\\\]\s*)'

        # 使用 re.split() 分割文本，因为模式有捕获组，所以匹配到的分隔符也会保留在结果列表中
        parts = re.split(math_formula_pattern, cleaned_string, flags=re.DOTALL)

        results = []
        for part in parts:
            part_stripped = part.strip() # 先去除当前片段的空白符

            if not part_stripped:
                continue # 跳过完全空白的片段

            # 判断是否是数学公式块
            if part_stripped.startswith('\\[') and part_stripped.endswith('\\]'):
                # 这是一个数学公式，移除 \[ 和 \]
                formula_content = part_stripped[2:-2].strip()
                results.append({"type": "formula", "content": formula_content})
            else:
                # 这是普通文本
                results.append({"type": "text", "content": part_stripped})

        return results

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
