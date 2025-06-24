import numpy as np
import wave
import pyaudio

def read_wav_data_as_float(file_path):
    """
    读取 WAV 文件，并将其音频数据转换为归一化到 [-1.0, 1.0] 的 float32 类型。
    返回音频参数和浮点音频数据。
    """
    try:
        with wave.open(file_path, 'rb') as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            audio_bytes = wf.readframes(n_frames)

            # 根据采样宽度确定原始数据类型和最大值
            if sample_width == 1: # 8-bit unsigned integer, range [0, 255] -> normalized to [-1.0, 1.0]
                dtype = np.uint8
                max_val = 128.0 # For 8-bit unsigned, center is 128, range is 128
                offset = 128.0
            elif sample_width == 2: # 16-bit signed integer, range [-32768, 32767]
                dtype = np.int16
                max_val = 32768.0 # Using 32768 to cover full range for signed int
                offset = 0.0
            elif sample_width == 4: # 32-bit signed integer, range [-2147483648, 2147483647]
                dtype = np.int32
                max_val = 2147483648.0
                offset = 0.0
            else:
                raise ValueError(f"Unsupported sample width: {sample_width} bytes")

            # 将字节数据转换为原始整数 NumPy 数组
            audio_data_int = np.frombuffer(audio_bytes, dtype=dtype)

            # 转换为 float32 并归一化到 [-1.0, 1.0]
            # 对于 8-bit unsigned，需要先减去中心值再归一化
            audio_data_float = (audio_data_int.astype(np.float32) - offset) / max_val
            
            # 如果是多通道，重塑为 (N_samples, N_channels)
            if n_channels > 1:
                audio_data_float = audio_data_float.reshape(-1, n_channels)

            return n_channels, sample_width, framerate, audio_data_float

    except Exception as e:
        print(f"读取 WAV 文件时出错: {e}")
        return None, None, None, None

def mix_stereo_to_mono_low_loss(
    stereo_float_data: np.ndarray,
    weights: tuple[float, float] = (0.5, 0.5)
) -> np.ndarray:
    """
    将立体声 (2 通道) 浮点音频数据合并为单声道 (1 通道)，
    旨在尽可能降低损耗。

    Args:
        stereo_float_data (np.ndarray): 立体声音频数据，归一化到 [-1.0, 1.0] 的 float32。
                                        预期形状为 (N_samples, 2)。
        weights (tuple[float, float]): 左右声道的混音权重。默认 (0.5, 0.5) 表示平均。
                                      注意：权重之和不建议超过 1.0，否则可能再次引入削波风险。

    Returns:
        np.ndarray: 合并后的单声道浮点音频数据，形状为 (N_samples,)。
                    同样归一化到 [-1.0, 1.0]。
    """
    if stereo_float_data.ndim != 2 or stereo_float_data.shape[1] != 2:
        raise ValueError("输入数据必须是形状为 (N_samples, 2) 的立体声浮点数据。")
    if not (0 <= weights[0] <= 1 and 0 <= weights[1] <= 1):
         raise ValueError("权重必须在 0 到 1 之间。")

    left_channel = stereo_float_data[:, 0]
    right_channel = stereo_float_data[:, 1]

    # 加权平均合并
    mono_float_data = weights[0] * left_channel + weights[1] * right_channel

    # 简单加权平均（如 0.5, 0.5）通常不会导致超出 [-1.0, 1.0] 范围，
    # 但如果权重之和超过 1.0 或者原始信号有特殊情况，可能需要额外的钳位或动态调整。
    # 为了鲁棒性，总是进行一次钳位是好习惯。
    mono_float_data = np.clip(mono_float_data, -1.0, 1.0)

    return mono_float_data

def play_audio_data_from_float(channels, framerate, audio_float_data, original_sample_width, chunk_size=1024):
    """
    播放归一化后的浮点音频数据，并将其转换回原始的整数格式。
    """
    p = pyaudio.PyAudio()

    # 确定输出到 PyAudio 的格式
    if original_sample_width == 1:
        pyaudio_format = pyaudio.paInt8
        max_val = 128.0
        offset = 128
        target_dtype = np.uint8
    elif original_sample_width == 2:
        pyaudio_format = pyaudio.paInt16
        max_val = 32768.0
        offset = 0
        target_dtype = np.int16
    elif original_sample_width == 4:
        pyaudio_format = pyaudio.paInt32
        max_val = 2147483648.0
        offset = 0
        target_dtype = np.int32
    else:
        raise ValueError(f"Unsupported sample width for playback: {original_sample_width} bytes")

    stream = p.open(format=pyaudio_format,
                    channels=channels,
                    rate=framerate,
                    output=True,
                    frames_per_buffer=chunk_size)

    print("正在播放合并后的音频...")

    # 将浮点数据转换回原始整数范围和类型
    # 乘以最大值，加上偏移量（如果原始是 unsigned int），四舍五入，然后钳位
    audio_data_for_playback = (audio_float_data * max_val + offset).round().astype(target_dtype)
    
    # 钳位确保不超出原始数据类型范围
    if np.issubdtype(target_dtype, np.integer):
        info = np.iinfo(target_dtype)
        audio_data_for_playback = np.clip(audio_data_for_playback, info.min, info.max)

    stream.write(audio_data_for_playback.tobytes())

    print("播放完成。")
    stream.stop_stream()
    stream.close()
    p.terminate()


# --- 主程序流程 ---
if __name__ == "__main__":
    wav_file_path = "output_multi_channel.wav" # 假设这是你的立体声 WAV 文件

    # 1. 读取 WAV 文件，并转换为归一化浮点数
    n_channels, sample_width_bytes, framerate, audio_data_float = read_wav_data_as_float(wav_file_path)

    if audio_data_float is not None:
        if n_channels != 2:
            print(f"警告: WAV 文件不是立体声 ({n_channels} 通道)。无法进行立体声混音。")
            # 如果不是立体声，直接播放原始数据或进行其他处理
            play_audio_data_from_float(n_channels, framerate, audio_data_float, sample_width_bytes)
        else:
            # 2. 合并左右声道
            # 使用默认的 (0.5, 0.5) 权重进行平均
            mono_float_data = mix_stereo_to_mono_low_loss(audio_data_float)

            print(f"合并后的单声道浮点数据形状: {mono_float_data.shape}")

            # 3. 播放合并后的单声道音频
            play_audio_data_from_float(1, framerate, mono_float_data, sample_width_bytes)

            # 可选：将合并后的单声道浮点数据保存为 WAV 文件
            # 需先将浮点数据转回整数类型再保存
            def save_mono_wav_from_float(file_path, audio_float_array, rate, original_sample_width):
                if original_sample_width == 1:
                    max_val = 128.0
                    offset = 128
                    target_dtype = np.uint8
                elif original_sample_width == 2:
                    max_val = 32768.0
                    offset = 0
                    target_dtype = np.int16
                elif original_sample_width == 4:
                    max_val = 2147483648.0
                    offset = 0
                    target_dtype = np.int32
                else:
                    raise ValueError(f"Unsupported sample width for saving: {original_sample_width} bytes")

                # 转换回原始整数类型并钳位
                audio_int_data = (audio_float_array * max_val + offset).round().astype(target_dtype)
                if np.issubdtype(target_dtype, np.integer):
                    info = np.iinfo(target_dtype)
                    audio_int_data = np.clip(audio_int_data, info.min, info.max)
                
                with wave.open(file_path, 'wb') as wf:
                    wf.setnchannels(1) # 单声道
                    wf.setsampwidth(original_sample_width)
                    wf.setframerate(rate)
                    wf.writeframes(audio_int_data.tobytes())
                print(f"单声道 WAV 文件已保存到: {file_path}")
            
            # save_mono_wav_from_float("mono_output_low_loss.wav", mono_float_data, framerate, sample_width_bytes)