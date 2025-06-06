# ------------------------------------------------------------
# 1.  Setup: imports & model
# ------------------------------------------------------------
# from transformers import AutoTokenizer, AutoModelForVisionText2Text
import torch, textwrap, pprint
import utils
from transformers import AutoModel, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from conversation import get_conv_template  # You'll need to import this
import utils_video

hf_path = "OpenGVLab/InternVL3-1B-Pretrained"      # change if you use a different size
device = "cuda" if torch.cuda.is_available() else "cpu"
hf_path = utils.get_mvl_model()
hf_path = "./custom_models/InternVL3-1B-Pretrained"
chunk_path = "/home/bagga005/algo/comp_data/video_chunks/s3/friends_s03e06a_tr_3.mp4"
question_for_embeddings = "| Scene: Central Perk, the whole gang is entering | \
Joey: I'm tellin' ya that girl ... \
Frame1: <image> \
Frame2: <image> \
Frame3: <image> \
Frame4: <image> \
Frame5: <image> \
Frame6: <image> \
Frame7: <image> \
Frame8: <image>\
\
| Dialogue |\
... totally winked at me.\
All: Did not, she did not ..."

question_for_embeddings_v2 = "| Scene: Central Perk, the whole gang is entering | \
Joey: I'm tellin' ya that girl ... \
| Dialogue |\
... totally winked at me.\
All: Did not, she did not ..."

# Your text prompt
text_prompt = question_for_embeddings_v2


#print(question_for_embeddings)

    # For the second model loading instance
    # Configure quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False
)
tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True, use_fast=True)
# Add this line to set pad_token_id
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'left'
question_list = [question_for_embeddings, question_for_embeddings_v2]
model_inputs = tokenizer(question_list, return_tensors='pt', return_offsets_mapping=True, padding=True)
input_ids = model_inputs['input_ids'].to(device)
print("input_ids", input_ids.shape)
model_inputs = tokenizer(question_for_embeddings, return_tensors='pt', return_offsets_mapping=True)
input_ids = model_inputs['input_ids'].to(device)
print("input_ids", input_ids.shape)
model_inputs = tokenizer(question_for_embeddings_v2, return_tensors='pt', return_offsets_mapping=True)
input_ids = model_inputs['input_ids'].to(device)
print("input_ids", input_ids.shape)
exit()
model = AutoModel.from_pretrained(
        hf_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True)


# pixel_values, num_patches_list = utils_video.load_video(chunk_path, num_segments=8, max_num=1)
# pixel_values = pixel_values.to(torch.bfloat16).cuda()
pixel_values = torch.randn(16, 3, 448, 448, dtype=torch.bfloat16, device=model.device)
print('pixel_values.shape', pixel_values.shape)

#setup for forward and generate
# one dummy 448×448 RGB frame (ViT default size); replace with real images
# pixel_values = torch.randn(1, 3, 448, 448, dtype=torch.bfloat16, device=device)

# # # Define special tokens
# IMG_START_TOKEN = '<img>'
# IMG_END_TOKEN = '</img>'
# IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'

# # # Get the IMG_CONTEXT_TOKEN id
# img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
# model.img_context_token_id = img_context_token_id
# model_inputs = tokenizer(text_prompt, return_tensors='pt')
# input_ids = model_inputs['input_ids'].to(device)
# print("input_ids", input_ids.shape)

# # # Set up conversation template
# template = get_conv_template(model.template)

# # # Add the prompt to the template (if it doesn't have <image>, add it)
# if '<image>' not in text_prompt:
#     text_prompt = '<image>\n' + text_prompt

# template.append_message(template.roles[0], text_prompt)
# template.append_message(template.roles[1], None)
# query = template.get_prompt()
# print("query", query)

# # # Replace <image> with the actual image tokens
# # # num_patches = 1 for one image
# num_patches = 1
# image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
# query = query.replace('<image>', image_tokens, 1)

# # # Tokenize the query
# model_inputs = tokenizer(query, return_tensors='pt')
# input_ids = model_inputs['input_ids'].to(device)
# print("input_ids", input_ids.shape)
# attention_mask = model_inputs['attention_mask'].to(device)

# # # Create image_flags tensor (1 for each image that should be processed)
# # # Since we have 1 image, image_flags should be a tensor with a single 1
# image_flags = torch.ones(pixel_values.shape[0], dtype=torch.long, device=device)


# gen_out = model.generate(
#     **tokenizer(query, return_tensors="pt").to(device),
#     pixel_values            = pixel_values.to(device),
#     max_new_tokens          = 100,
#     output_hidden_states    = True,
#     return_dict_in_generate = True,   # valid here
# )
# seq = gen_out.sequences
# print(seq[0][-1])
# for i in range(len(seq[0])):
#     print(tokenizer.decode(seq[0][i]))

# Call forward method
# with torch.no_grad():
#     outputs = model(
#         pixel_values=pixel_values,
#         input_ids=input_ids,
#         #attention_mask=attention_mask,
#         image_flags=image_flags,
#         return_dict=True,
#         output_hidden_states  = True
#     )
# #Get logits
# logits = outputs.logits
# print(outputs.hidden_states[-1].shape)
# print("Forward pass completed. Logits shape:", logits.shape)

#call chat method
generation_config = dict(
    max_new_tokens=1000,
    pad_token_id=tokenizer.pad_token_id  # Explicitly set to avoid warning
)
tokenizer.padding_side = 'left'
question_list = [text_prompt, text_prompt]
model_inputs = tokenizer(question_list, return_tensors='pt', return_offsets_mapping=True)
input_ids = model_inputs['input_ids'].to(device)
print("input_ids", input_ids.shape)
print('calling batch')
num_patches_list = list(range(0, 2))

# model.batch_chat(
#     tokenizer, 
#     pixel_values, 
#     question_list, 
#     generation_config,
#     num_patches_list=num_patches_list,
#     history=None, 
#     return_history=False,
#     verbose=True,
#     #output_hidden_states=True
# )