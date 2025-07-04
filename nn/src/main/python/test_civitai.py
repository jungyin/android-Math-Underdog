import os

# 设置环境变量
os.environ['CIVITAI_API_TOKEN'] = '9fe03e1e5eddc5bf1b3feb382a4eb7ff'

import civitai

urn = "urn:civitai:modelVersion:"

input = {
    "model": "urn:air:sd1:checkpoint:civitai:4201@130072",
    "params": {
        "prompt": "RAW photo, face portrait photo of 26 y.o woman, wearing black dress, happy face, hard shadows, cinematic shot, dramatic lighting",
        "negativePrompt": "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3)",
        "scheduler": "EulerA",
        "steps": 20,
        "cfgScale": 7,
        "width": 512,
        "height": 512,
        "clipSkip": 2
    }
}

response = civitai.image.create(input)
print(response)