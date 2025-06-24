from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


# 音频降噪 https://www.modelscope.cn/models/iic/speech_frcrn_ans_cirm_16k
ans = pipeline(
    Tasks.acoustic_noise_suppression,
    model='iic/speech_frcrn_ans_cirm_16k')
result = ans(
    'https://modelscope.oss-cn-beijing.aliyuncs.com/test/audios/speech_with_noise1.wav',
    output_path='output.wav')
print( result)