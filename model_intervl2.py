from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig
from lmdeploy.vl import load_image


model = "OpenGVLab/InternVL3-1B-Pretrained"
image = load_image('https://www.cnet.com/a/img/resize/31b06f1bbd53211efcab1045e86e0443ba3f894e/hub/2025/05/13/581070e4-da9e-44d4-a998-002c03762556/gettyimages-2214102949.jpg?auto=webp&fit=crop&height=675&width=1200')
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=64, tp=1), chat_template_config=ChatTemplateConfig(model_name='internvl2_5'))
response = pipe(('describe this image', image))
print(response.text)