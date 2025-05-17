from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig
from lmdeploy.vl import load_image


model = "OpenGVLab/InternVL3-1B-Pretrained"
image = load_image('https://techcrunch.com/wp-content/uploads/2025/02/GettyImages-2197091379.jpg')
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=64, tp=1), chat_template_config=ChatTemplateConfig(model_name='internvl2_5'))
response = pipe(('describe this image', image))
print(response.text)