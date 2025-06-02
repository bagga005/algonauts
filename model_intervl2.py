# ------------------------------------------------------------
# 1.  Setup: imports & model
# ------------------------------------------------------------
# from transformers import AutoTokenizer, AutoModelForVisionText2Text
import torch, textwrap, pprint
import utils
from transformers import AutoModel, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from conversation import get_conv_template  # You'll need to import this

hf_path = "OpenGVLab/InternVL3-1B-Pretrained"      # change if you use a different size
device = "cuda" if torch.cuda.is_available() else "cpu"
hf_path = utils.get_mvl_model()
hf_path = "./custom_models/InternVL3-1B-Pretrained"

    # For the second model loading instance
    # Configure quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False
)


model = AutoModel.from_pretrained(
        hf_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True, use_fast=True)

# one dummy 448×448 RGB frame (ViT default size); replace with real images
pixel_values = torch.randn(1, 3, 448, 448, dtype=torch.bfloat16, device=device)

# Define special tokens
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'

# Get the IMG_CONTEXT_TOKEN id
img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
model.img_context_token_id = img_context_token_id

# Your text prompt
text_prompt = '| Scene: Central Perk, the whole gang is there including Janice. |\nFrame1: <image>  More text'

# Set up conversation template
template = get_conv_template(model.template)
template.system_message = model.system_message

# Add the prompt to the template (if it doesn't have <image>, add it)
if '<image>' not in text_prompt:
    text_prompt = '<image>\n' + text_prompt

template.append_message(template.roles[0], text_prompt)
template.append_message(template.roles[1], None)
query = template.get_prompt()

# Replace <image> with the actual image tokens
# num_patches = 1 for one image
num_patches = 1
image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
query = query.replace('<image>', image_tokens, 1)

# Tokenize the query
model_inputs = tokenizer(query, return_tensors='pt')
input_ids = model_inputs['input_ids'].to(device)
attention_mask = model_inputs['attention_mask'].to(device)

# Create image_flags tensor (1 for each image that should be processed)
# Since we have 1 image, image_flags should be a tensor with a single 1
image_flags = torch.ones(pixel_values.shape[0], dtype=torch.long, device=device)


gen_out = model.generate(
    **tokenizer(query, return_tensors="pt").to(device),
    pixel_values            = pixel_values.to(device),
    max_new_tokens          = 100,
    output_hidden_states    = True,
    return_dict_in_generate = True,   # valid here
)
seq = gen_out.sequences
print(seq[0][-1])
for i in range(len(seq[0])):
    print(tokenizer.decode(seq[0][i]))

# Call forward method
# with torch.no_grad():
#     outputs = model(
#         pixel_values=pixel_values,
#         input_ids=input_ids,
#         attention_mask=attention_mask,
#         image_flags=image_flags,
#         return_dict=True,
#         output_hidden_states  = True
#     )
# Get logits
#logits = outputs.logits
#print(gen_out.hidden_states[-1])
# print("Forward pass completed. Logits shape:", logits.shape)