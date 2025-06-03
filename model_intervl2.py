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
chunk_path = "/workspace/algo_data/video_chunks/s3/friends_s03e01a_tr_3.mp4"
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
Frame1: <image> \
\
| Dialogue |\
... totally winked at me.\
All: Did not, she did not ..."

# Your text prompt
text_prompt = question_for_embeddings
#print(question_for_embeddings)

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
# Add this line to set pad_token_id
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

pixel_values, num_patches_list = utils_video.load_video(chunk_path, num_segments=8, max_num=1)
pixel_values = pixel_values.to(torch.bfloat16).cuda()

print('pixel_values.shape', pixel_values.shape)

#setup for forward and generate
# one dummy 448×448 RGB frame (ViT default size); replace with real images
#pixel_values = torch.randn(1, 3, 448, 448, dtype=torch.bfloat16, device=device)

# # Define special tokens
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'



#Phase 1 tokens
print("****Phase 1 tokens")
model_inputs = tokenizer(text_prompt, return_tensors='pt', return_offsets_mapping=True)
input_ids = model_inputs['input_ids'].to(device)
offsets = model_inputs.offset_mapping
print("input_ids (phase 1)", input_ids.shape)
print("offsets (phase 1)", offsets.shape)

print("****Phase 2 tokens")
# # Set up conversation template
template = get_conv_template(model.template)

# # Add the prompt to the template (if it doesn't have <image>, add it)
if '<image>' not in text_prompt:
    text_prompt = '<image>\n' + text_prompt
template.append_message(template.roles[0], text_prompt)
template.append_message(template.roles[1], None)
query = template.get_prompt()
model_inputs = tokenizer(query, return_tensors='pt', return_offsets_mapping=True)
input_ids = model_inputs['input_ids'].to(device)
offsets = model_inputs.offset_mapping
print("input_ids (phase 2)", input_ids.shape)
print("offsets (phase 2)", offsets.shape)


# Phase 3 add IMG_CONTEXT_TOKEN
num_patches = 1
image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token  + IMG_END_TOKEN
query = query.replace('<image>', image_tokens, pixel_values.shape[0])

# # Tokenize the query
model_inputs = tokenizer(query, return_tensors='pt', return_offsets_mapping=True)
input_ids = model_inputs['input_ids'].to(device)
offsets = model_inputs.offset_mapping
print("input_ids (phase 3)", input_ids.shape)
print("offsets (phase 3)", offsets.shape)
#utils.print_input_tokens_with_offsets(query, offsets, input_ids)
#attention_mask = model_inputs['attention_mask'].to(device)

# # Create image_flags tensor (1 for each image that should be processed)
# # Since we have 1 image, image_flags should be a tensor with a single 1
img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
model.img_context_token_id = img_context_token_id
img_start_token_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
img_end_token_id = tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
print("img_start_token_id", img_start_token_id)
print("img_end_token_id", img_end_token_id)
print("img_context_token_id", img_context_token_id)

def get_img_index_values(input_ids, img_start_token_id, img_end_token_id):
#populate key index values from inputs
    img_index_values = []
    start_index = None
    for i in range(input_ids.shape[1]):
        if input_ids[0, i] == img_start_token_id:
            start_index = i
        elif input_ids[0, i] == img_end_token_id:
            img_index_values.append((start_index, i))
            start_index = None
    return img_index_values

img_index_values = get_img_index_values(input_ids, img_start_token_id, img_end_token_id)
print("img_index_values", img_index_values)

def check_img_index_values(img_index_values):
    assert len(img_index_values) == pixel_values.shape[0]
    for i in range(len(img_index_values)):
        assert img_index_values[i][1] - img_index_values[i][0] == model.num_image_token +1

def get_pre_post_index_values(input_ids, tokenzier, img_index_values):
    pre_start_index = None
    pre_end_index = None
    post_start_index = None
    post_end_index = None
    last_chat_end_index = None
    image_end_index = None
    chat_start_word = '<|im_start|>'
    chat_end_word = '<|im_end|>'
    dialogue_word = ' Dialogue'
    dialogue_token_id = tokenizer.convert_tokens_to_ids(dialogue_word)
    print("dialogue_token_id", dialogue_token_id)
    chat_start_token_id = tokenizer.convert_tokens_to_ids(chat_start_word)
    chat_end_token_id = tokenizer.convert_tokens_to_ids(chat_end_word)
    chat_start_index_values = []
    chat_end_index_values = []
    for i in range(input_ids.shape[1]):
        if input_ids[0, i] == chat_start_token_id:
            chat_start_index_values.append(i)
        elif input_ids[0, i] == chat_end_token_id:
            chat_end_index_values.append(i)
    assert len(chat_end_index_values) == 2, f"len(chat_end_index_values) {len(chat_end_index_values)} != 2"
    assert len(chat_start_index_values) == 3, f"len(chat_start_index_values) {len(chat_start_index_values)} != 3"
    last_chat_end_index = chat_end_index_values[1]
    #4 tokens after start of second chat
    pre_start_index = chat_start_index_values[1] + 4
    #4 tokens before the start of first image
    pre_end_index = img_index_values[0][0] - 5
    #4 chars after the last image end
    post_start_index = img_index_values[len(img_index_values)-1][1] + 4
    #1 char before the last chat end
    post_end_index = chat_end_index_values[1] - 1
    return pre_start_index, pre_end_index, post_start_index, post_end_index, last_chat_end_index

check_img_index_values(img_index_values)
pre_start_index, pre_end_index, post_start_index, post_end_index, last_chat_end_index = get_pre_post_index_values(input_ids, tokenizer, img_index_values)
#utils.print_input_tokens_with_offsets(query, offsets, input_ids, pre_start=pre_start_index, pre_end=pre_end_index, post_start=post_start_index, post_end=post_end_index, last_chat_end=last_chat_end_index)
# Call forward method

#Get logits
#print("outputs", outputs)
# logits = outputs.logits
#print(outputs.hidden_states[-1].shape)
# print("Forward pass completed. Logits shape:", logits.shape)

import os

def create_layer_hooks(model, layers=None):
    """
    Create and return hooks for specified layers along with storage for outputs
    
    Args:
        model: The InternVL model
        layers: List of layer names to extract from
        
    Returns:
        tuple of (hooks list, layer_outputs dict, hook_storage dict)
    """
    if layers is None:
        layers = [
            'vision_model',
            'mlp1',
            'language_model.model.layers.0',
            'language_model.model.layers.2',
            'language_model.model.layers.5', 
            'language_model.model.layers.10',
            'language_model.model.norm'
        ]
    
    hooks = []
    hook_storage = {}  # Will store the actual hook functions
    layer_outputs = {}  # Will store the outputs
    
    def get_hook(name):
        def hook(module, input, output):
            if hasattr(output, 'last_hidden_state'):
                layer_outputs[name] = output.last_hidden_state.detach()
            elif isinstance(output, tuple):
                layer_outputs[name] = tuple(x.detach() if torch.is_tensor(x) else x for x in output)
            elif torch.is_tensor(output):
                layer_outputs[name] = output.detach()
            else:
                layer_outputs[name] = output
        return hook
    
    for layer_name in layers:
        if '.' in layer_name:
            parts = layer_name.split('.')
            module = model
            for part in parts:
                module = getattr(module, part)
            hook_fn = get_hook(layer_name)
            hook_storage[layer_name] = hook_fn
            hooks.append(module.register_forward_hook(hook_fn))
        else:
            hook_fn = get_hook(layer_name)
            hook_storage[layer_name] = hook_fn
            hooks.append(getattr(model, layer_name).register_forward_hook(hook_fn))
    
    return hooks, layer_outputs, hook_storage

def save_embeddings(embeddings, save_dir, text="", prefix="", counter=0):
    """
    Save embeddings to files.
    
    Args:
        embeddings: Dictionary of layer names to embeddings
        save_dir: Directory to save the embeddings
        prefix: Optional prefix for the filenames (e.g., image name or ID)
        use_numpy: If True, save as numpy arrays. If False, save as PyTorch tensors
    """
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a metadata dictionary to store shapes and types
    metadata = {}
    
    for layer_name, embedding in embeddings.items():
        # Create a safe filename from the layer name
        safe_name = layer_name.replace('.', '_').replace('/', '_')
        if prefix:
            safe_name = f"{prefix}_{safe_name}"
            
        file_ext = ".pt.gz"
        
        if isinstance(embedding, tuple):
            # Handle tuple by taking first element
            embedding = embedding[0]

        # Save single tensor
        if not torch.is_tensor(embedding):
            print('not single tensor')
            embedding = torch.tensor(embedding)
        print(counter,layer_name, embedding.shape)
        if 'language' in layer_name:
            #print('language', embedding.shape)
            #embedding = embedding.squeeze(0)
            #embedding = embedding[-10:,:]
            #print average of embedding
            print(f'{layer_name} average of embedding batch 0', embedding[0,:,:].mean())
            if embedding.shape[0] > 1:
                print(f'{layer_name} average of embedding batch 1', embedding[1,:,:].mean())
            #print('language', embedding.shape)
        if 'vision' in layer_name:
            #print('vision', embedding.shape)
            embedding = embedding[:,0,:]
            #print('vision', embedding.shape)
        # with gzip.open(os.path.join(save_dir, safe_name + file_ext), 'wb') as f:
        #     pickle.dump(embedding.cpu(), f)

        # with h5py.File(os.path.join(save_dir, safe_name + file_ext), 'w') as f:
        #     f.create_dataset('data', data=embedding.numpy())#, compression="gzip")
        metadata[layer_name] = {
            'type': 'tensor',
            'shape': list(embedding.shape) if hasattr(embedding, 'shape') else None
        }
    
    metadata['text'] = text
    # Save metadata
    # with open(os.path.join(save_dir, f"{prefix}_metadata.json" if prefix else "metadata.json"), 'w') as f:
    #     json.dump(metadata, f, indent=2)


custom_layers = [
            'vision_model.encoder.layers.2',
            'vision_model.encoder.layers.5',
            'vision_model.encoder.layers.10',
            'vision_model.encoder.layers.17',
            'vision_model.encoder.layers.23',
            'vision_model',                     # Vision encoder
            'language_model.model.layers.0',    # First layer
            'language_model.model.layers.4',    # First layer
            'language_model.model.layers.8',    # First layer
            'language_model.model.layers.12',    # First layer
            'language_model.model.layers.16',    # Middle layer
            'language_model.model.layers.20',   # Later layer
            'language_model.model.layers.23',   # Later layer
            'language_model.model.norm'         # Final normalization
        ]
hooks, layer_outputs, hook_storage = create_layer_hooks(model, custom_layers)

def generate_attention_mask_batch(input_ids_list, pad_token_id, device, max_length=None):
    """
    Generate attention masks for a batch of sequences with different lengths.
    
    Args:
        input_ids_list: List of lists/tensors, each containing token ids
        pad_token_id: int, the token id used for padding
        max_length: int, optional max length to pad to (if None, uses longest sequence)
    
    Returns:
        input_ids_padded: torch.Tensor [batch_size, max_seq_len] - padded input_ids
        attention_mask: torch.Tensor [batch_size, max_seq_len] - corresponding attention mask
    """
    import torch
    
    # Convert all to tensors
    input_ids_tensors = input_ids_list
    # for ids in input_ids_list:
    #     if isinstance(ids, list):
    #         ids = torch.tensor(ids)
    #     input_ids_tensors.append(ids)
    
    # Find max length
    if max_length is None:
        max_length = max(len(ids) for ids in input_ids_tensors)
    print("max_length", max_length)
    # Pad sequences and create attention masks
    batch_input_ids = []
    batch_attention_mask = []
    
    for ids in input_ids_tensors:
        current_length = len(ids)
        
        if current_length < max_length:
            # Pad if shorter than max_length
            padding_length = max_length - current_length
            paddings = torch.full((padding_length,), pad_token_id, dtype=ids.dtype, device=device)
            print("paddings", paddings.shape)
            print("ids", ids.shape)
            ids_padded = torch.cat([paddings, ids])
            attention = torch.cat([torch.zeros(padding_length, dtype=torch.long, device=device),
                                   torch.ones(current_length, dtype=torch.long, device=device)])
            ids = ids_padded
        else:
            attention = torch.ones(max_length, dtype=torch.long, device=device)
        batch_input_ids.append(ids)
        batch_attention_mask.append(attention)
    
    return torch.stack(batch_input_ids), torch.stack(batch_attention_mask)



input_ids_2 = input_ids.detach().clone().to(device)
input_ids_2 = input_ids_2[:,40:]
input_ids_list = [input_ids.squeeze(0), input_ids_2.squeeze(0)]
input_ids_padded, attention_mask = generate_attention_mask_batch(input_ids_list, pad_token_id=tokenizer.pad_token_id)
print("input_ids_padded shape", input_ids_padded.shape)
print("attention_mask shape", attention_mask.shape)
print("input_ids_padded", input_ids_padded)
print("attention_mask", attention_mask)

print("pixel_values", pixel_values.shape)
pixel_values_2 = pixel_values.detach().clone().to(device)
pixel_values = torch.cat((pixel_values, pixel_values_2), dim=0)
print("pixel_values", pixel_values.shape)
image_flags = torch.ones(pixel_values.shape[0], dtype=torch.long, device=device) 
with torch.no_grad():
    outputs = model(
        pixel_values=pixel_values,
        input_ids=input_ids_padded,
        image_flags=image_flags,
        return_dict=False,
        output_hidden_states  = False,
        attention_mask=attention_mask
    )

save_embeddings(layer_outputs, "embeddings4")
       