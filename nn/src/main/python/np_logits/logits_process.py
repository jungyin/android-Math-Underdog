import numpy as np

from nputils import softmax,masked_fill

class RepetitionPenaltyLogitsProcessor():
    def __init__(self, penalty: float):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = penalty
    def __call__(self, input_ids, scores):
        np.take_along_axis(scores, input_ids, axis=1)
        # if score < 0 then repetition penalty has to be multiplied to reduce the token probabilities
        score = np.where(scores < 0, scores * self.penalty, scores / self.penalty)
       # 初始化一个与'scores'相同大小的全零数组作为掩码
        mask = np.zeros_like(scores)
        # 根据input_ids设置掩码位置为1
        np.put_along_axis(mask, input_ids, 1, axis=1)
        # 更新'scores'，仅在mask为1的位置应用新的'score'值
        scores_processed = np.where(mask == 1, score, scores)
    
        return scores_processed

class TemperatureLogitsWarper():

    def __init__(self, temperature: float):
        if not isinstance(temperature, float) or not (temperature > 0):
            except_msg = (
                f"`temperature` (={temperature}) has to be a strictly positive float, otherwise your next token "
                "scores will be invalid."
            )
            if isinstance(temperature, float) and temperature == 0.0:
                except_msg += " If you're looking for greedy decoding strategies, set `do_sample=False`."
            raise ValueError(except_msg)

        self.temperature = temperature

    def __call__(self, input_ids, scores):
        scores_processed = scores / self.temperature
        return scores_processed


class TopPLogitsWarper():

    def __init__(self, top_p: float, filter_value: float = -np.inf, min_tokens_to_keep: int = 1):
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1): 
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")

        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids, scores) :
        sorted_logits = np.sort(scores)
        sorted_indices = np.argsort(scores)
        cumulative_probs = softmax(sorted_logits,-1)
        cumulative_probs = np.cumsum(cumulative_probs, axis=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., -self.min_tokens_to_keep :] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove= np.zeros_like(sorted_indices_to_remove,dtype=bool)

        np.put_along_axis(indices_to_remove,sorted_indices,sorted_indices_to_remove,axis=1)
        
        scores_processed = masked_fill(scores,indices_to_remove, self.filter_value)
        return scores_processed


class TopKLogitsWarper():
    def __init__(self, top_k: int, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")

        self.top_k = max(top_k, min_tokens_to_keep)
        self.filter_value = filter_value

    def __call__(self, input_ids, scores) :
        top_k = min(self.top_k, scores.shape[-1])  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k

        # 找到每个子数组中最大的top_k个值的位置
        partitioned_indices = np.argpartition(-scores,top_k - 1,axis=-1)

        # 提取每个子数组中前top_k个最大值
        top_k_values = np.take_along_axis(scores, partitioned_indices[:, :top_k], axis=1)

        # 对每个子数组中的值进行排序以确保它们按降序排列
        sorted_indices = np.argsort(-top_k_values, axis=1)
        sorted_top_k_values = np.take_along_axis(top_k_values, sorted_indices, axis=1)

        # 获取第top_k大的值
        last_of_top_k = sorted_top_k_values[:, -1:]
        last_of_top_k = last_of_top_k[..., -1, np.newaxis]
        # 添加新维度使其成为列向量
        indices_to_remove = scores < last_of_top_k

        scores_processed = masked_fill(scores,indices_to_remove,self.filter_value)
        return scores_processed


class EosTokenCriteria():
    def __init__(self, eos_token_id):
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        self.eos_token_id = np.array(eos_token_id)
    def __call__(self, input_ids, scores) :
        is_done = np.isin(input_ids[:, -1], self.eos_token_id)
        return is_done


class MaxLengthCriteria():
    def __init__(self, max_length: int, max_position_embeddings = None):
        self.max_length = max_length
        self.max_position_embeddings = max_position_embeddings
    def __call__(self, input_ids, scores) :
        cur_len = input_ids.shape[-1]
        is_done = cur_len >= self.max_length
        if self.max_position_embeddings is not None and not is_done and cur_len >= self.max_position_embeddings:
            print(
                "This is a friendly reminder - the current text generation call will exceed the model's predefined "
                f"maximum length ({self.max_position_embeddings}). Depending on the model, you may observe "
                "exceptions, performance degradation, or nothing at all."
            )
        return np.full((input_ids.shape[0],), is_done)

