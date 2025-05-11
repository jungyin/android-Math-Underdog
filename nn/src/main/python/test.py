import openvino_genai as ov_genai
model_path = "D:\\code\\transformer_models\\models--Qwen--Qwen2.5-3B-Instruct"
model_path = "TinyLlama"
pipe = ov_genai.LLMPipeline(model_path, "NPU")
print(pipe.generate("The Sun is yellow because", max_new_tokens=100, do_sample=False))