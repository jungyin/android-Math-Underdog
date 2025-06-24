import pyaudio
import wave

# 音频参数
FORMAT = pyaudio.paInt16  # 16-bit int 格式
CHANNELS = 1              # 单声道
RATE = 44100              # 采样率 (Hz)
CHUNK = 1024              # 每次读取的帧数
RECORD_SECONDS = 5        # 录制时长 (秒)
WAVE_OUTPUT_FILENAME = "output.wav" # 输出文件名

audio = pyaudio.PyAudio()

# 查找默认输入设备
# 或者你可以手动指定 device_index
info = audio.get_host_api_info_by_index(0)
num_devices = info.get('deviceCount')
default_input_device_index = None
print("--- 可用音频输入设备 ---")
for i in range(num_devices):
    device_info = audio.get_device_info_by_host_api_device_index(0, i)
    if device_info.get('maxInputChannels') > 0:
        print(f"ID: {device_info.get('index')}, 名称: {device_info.get('name')}, 输入通道: {device_info.get('maxInputChannels')}")
        if device_info.get('isCurrentDefaultInput'): # 检查是否是默认输入设备
            default_input_device_index = device_info.get('index')
        elif default_input_device_index is None:
            default_input_device_index = device_info.get('index')
print("-----------------------")

if default_input_device_index is None:
    print("未找到默认输入设备，请手动指定 device_index。尝试使用 ID 0。")
    default_input_device_index = 0 # 备用方案

# 打开音频流
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=default_input_device_index)

print("正在录音...")

frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("录音结束。")

# 停止并关闭流
stream.stop_stream()
stream.close()
audio.terminate()

# 保存录音文件
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(audio.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

print(f"录音已保存到 {WAVE_OUTPUT_FILENAME}")