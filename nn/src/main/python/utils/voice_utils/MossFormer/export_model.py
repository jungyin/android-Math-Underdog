# MossFormer2语音分离模型 https://www.modelscope.cn/models/iic/speech_mossformer2_separation_temporal_8k

import numpy
import soundfile as sf
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# input可以是url也可以是本地文件路径
input = 'https://modelscope.cn/api/v1/models/damo/speech_mossformer2_separation_temporal_8k/repo?Revision=master&FilePath=examples/mix_speech1.wav'
separation = pipeline(
   Tasks.speech_separation,
   model='iic/speech_mossformer2_separation_temporal_8k')
result = separation(input)
for i, signal in enumerate(result['output_pcm_list']):
    save_file = f'output_spk{i}.wav'
    sf.write(save_file, numpy.frombuffer(signal, dtype=numpy.int16), 8000)