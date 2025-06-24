from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import torch 
from modelscope.models.audio.kws.nearfield.cmvn import  load_kaldi_cmvn


path = "C:/Users/30585/.cache/modelscope/hub/models/iic/speech_charctc_kws_phone-xiaoyun/"

kwsbp_16k_pipline = pipeline(
    task=Tasks.keyword_spotting,
    # model='iic/speech_charctc_kws_phone-xiaoyun')
    model=path,device='cpu',training=False)
    # model='iic/speech_charctc_kws_phone-xiaoyun',device='cpu')

tmodel = pipeline(
    task=Tasks.keyword_spotting,
    # model='iic/speech_charctc_kws_phone-xiaoyun')
    model=path,device='cpu',training=True)
model = tmodel.model
print(model)
weights= torch.load(path+"finetune_avg_10.pt")
model.load_state_dict(weights)
test = torch.ones([1,10000,400],device=kwsbp_16k_pipline.device)
out = model(test)
# rect = load_kaldi_cmvn(path + "am.mvn")

# model.eval()  # 设置为评估模式

kws_result =  kwsbp_16k_pipline(audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/KWS/pos_testset/kws_xiaoyunxiaoyun.wav')
kws_result = kwsbp_16k_pipline(audio_in='D:/code/android/android-Math-Underdog/nn/src/main/python/output.wav')
print(kws_result)


# kwsbp_16k_pipline = pipeline(
#     task=Tasks.keyword_spotting,
#     model='iic/speech_charctc_kws_phone-xiaoyun')

# kws_result = kwsbp_16k_pipline(audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/KWS/pos_testset/kws_xiaoyunxiaoyun.wav')
# print(kws_result)FSMNDecorator