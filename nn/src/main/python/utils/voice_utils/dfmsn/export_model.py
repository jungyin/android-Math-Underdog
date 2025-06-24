from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# DFSMN回声消除 https://www.modelscope.cn/models/iic/speech_dfsmn_aec_psm_16k/summary
input = {
    'nearend_mic': 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/audios/nearend_mic.wav',
    'farend_speech': 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/audios/farend_speech.wav'
}
aec = pipeline(
   Tasks.acoustic_echo_cancellation,
   model='iic/speech_dfsmn_aec_psm_16k')
result = aec(input, output_path='output.wav')
print(result)