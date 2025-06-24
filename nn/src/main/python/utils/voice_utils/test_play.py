import wave
import numpy as np

def read_wav_file(file_path):
    """
    读取 WAV 文件并返回音频参数和音频信号数据。

    Args:
        file_path (str): WAV 文件的路径。

    Returns:
        tuple: (n_channels, sample_width, framerate, n_frames, audio_data)
               其中 audio_data 是 NumPy 数组 (int16 格式)。
    """
    try:
        with wave.open(file_path, 'rb') as wf:
            n_channels = wf.getnchannels()     # 声道数
            sample_width = wf.getsampwidth()   # 采样宽度 (字节)
            framerate = wf.getframerate()      # 采样率
            n_frames = wf.getnframes()         # 帧数

            # 读取所有音频帧
            audio_bytes = wf.readframes(n_frames)

            # 将字节数据转换为 NumPy 数组
            # 注意：pyaudio.paInt16 对应 np.int16
            if sample_width == 1: # 8-bit int
                audio_data = np.frombuffer(audio_bytes, dtype=np.int8)
            elif sample_width == 2: # 16-bit int
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
            elif sample_width == 4: # 32-bit int or float (depends on WAV format)
                # For 32-bit float, use np.float32, for 32-bit int, use np.int32
                # Most common for speech is 16-bit int
                audio_data = np.frombuffer(audio_bytes, dtype=np.int32) # Assuming int32 for now
            else:
                raise ValueError(f"Unsupported sample width: {sample_width} bytes")

            print(f"文件: {file_path}")
            print(f"声道数: {n_channels}")
            print(f"采样宽度: {sample_width} 字节 ({sample_width * 8}-bit)")
            print(f"采样率: {framerate} Hz")
            print(f"帧数: {n_frames}")
            print(f"音频数据形状: {audio_data.shape}, 数据类型: {audio_data.dtype}")

            return n_channels, sample_width, framerate, n_frames, audio_data

    except wave.Error as e:
        print(f"读取 WAV 文件时出错: {e}")
        return None, None, None, None, None
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
        return None, None, None, None, None

# 使用之前录制的 output.wav
# 请确保你的 output.wav 或 output_multi_channel.wav 文件存在
# wav_file = "output.wav"
wav_file = "output_multi_channel.wav" # 如果你录制的是多通道

n_channels, sample_width, framerate, n_frames, audio_data = read_wav_file(wav_file)

if audio_data is not None:
    # 如果是多通道数据，它将是交错的。
    # 例如，对于 2 通道，audio_data 是 [L1, R1, L2, R2, ...]
    # 如果你想分别访问每个通道的数据：
    if n_channels > 1:
        # 将一维交错数据重塑为 (n_frames, n_channels)
        channel_data = audio_data.reshape(-1, n_channels)
        print(f"重塑后的通道数据形状: {channel_data.shape}")
        
        # 访问第一个通道的信号 (例如，左声道)
        first_channel_signal = channel_data[:, 0]
        print(f"第一个通道的信号形状: {first_channel_signal.shape}")
        
        # 访问第二个通道的信号 (例如，右声道)
        if n_channels >= 2:
            second_channel_signal = channel_data[:, 1]
            print(f"第二个通道的信号形状: {second_channel_signal.shape}")
        
        # 你可以通过 channel_data[:, i] 访问第 i 个通道的信号
    else:
        # 对于单通道，audio_data 就是整个通道的信号
        print(f"单通道信号形状: {audio_data.shape}")

    # 获取每一个音频信号（即每一个采样点的值）
    # audio_data 或 first_channel_signal 就是包含了每一个采样点的 NumPy 数组。
    # 你可以直接遍历这个数组来“检查”每个信号：
    print("\n--- 前10个音频信号值（第一个通道，如果存在）---")
    if n_channels > 1:
        print(first_channel_signal[:10])
    else:
        print(audio_data[:10])