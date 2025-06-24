import pyaudio
import numpy as np # 用于数据处理
import wave

FORMAT = pyaudio.paInt16
CHANNELS = 2          # 例如，立体声
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME_MULTI = "output_multi_channel.wav"

audio = pyaudio.PyAudio()

# (同上，查找并选择输入设备)
info = audio.get_host_api_info_by_index(0)
num_devices = info.get('deviceCount')
selected_device_index = None # 需要选择一个支持多通道的设备

print("--- 可用音频输入设备（多通道）---")
for i in range(num_devices):
    device_info = audio.get_device_info_by_host_api_device_index(0, i)
    # 查找支持至少 CHANNELS 个输入通道的设备
    if device_info.get('maxInputChannels') >= CHANNELS:
        print(f"ID: {device_info.get('index')}, 名称: {device_info.get('name')}, 输入通道: {device_info.get('maxInputChannels')}")
        # 这里你可以根据名称或ID选择你想要的多通道麦克风
        # 简单起见，我们选择第一个符合条件的
        if selected_device_index is None: # 找到第一个支持所需通道数的设备
             selected_device_index = device_info.get('index')
print("---------------------------------")

if selected_device_index is None:
    print(f"未找到支持 {CHANNELS} 个通道的设备，请检查你的麦克风设置。尝试使用默认设备ID 0。")
    selected_device_index = 0 # 备用方案

stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=selected_device_index)

print(f"正在录制 {CHANNELS} 通道音频...")

frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("录音结束。")

stream.stop_stream()
stream.close()
audio.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME_MULTI, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(audio.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

print(f"多通道录音已保存到 {WAVE_OUTPUT_FILENAME_MULTI}")

# 提取多通道数据
# 读取的数据是 interleaved 的。要提取单个通道，你需要将其分离。
# 比如对于 16-bit int，2通道：
audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
print(f"原始音频数据形状: {audio_data.shape}")

# 如果是立体声 (2通道)，数据是 [L1, R1, L2, R2, ...]
# 可以使用 reshape 或切片来分离通道
if CHANNELS == 2:
    channel_data = audio_data.reshape(-1, CHANNELS)
    left_channel = channel_data[:, 0]
    right_channel = channel_data[:, 1]
    print(f"左声道数据形状: {left_channel.shape}")
    print(f"右声道数据形状: {right_channel.shape}")
elif CHANNELS > 2:
    # 对于更多通道，你可以类似地分离
    channel_data = audio_data.reshape(-1, CHANNELS)
    channels_list = [channel_data[:, i] for i in range(CHANNELS)]
    print(f"分离后的通道数据形状（第一个通道）: {channels_list[0].shape}")