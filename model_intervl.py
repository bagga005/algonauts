import math
import numpy as np
import torch
import utils_video
from transformers import AutoModel, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from torchvision.models.feature_extraction import create_feature_extractor
import os
import json
import utils
from glob import glob
from tqdm import tqdm
import pandas as pd
from SentenceDataset import SentenceDataset_v2
import gzip
import pickle
from Scenes_and_dialogues import get_scene_dialogue
from transcripts_handler import load_all_tsv_for_one_episode
from tabulate import tabulate
from conversation import get_conv_template


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map
def save_embeddings(full_embeddings, prompt_markers_list, save_dir, text="", list_prefix=None, counter=0):
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
    file_ext = ".pt.gz"
    simple_extraction = utils.get_mvl_simple_extraction()
    assert len(prompt_markers_list) == len(list_prefix), f"len(prompt_markers_list) != len(list_prefix) {len(prompt_markers_list)} != {len(list_prefix)}"
    for i, (prompt_markers, prefix) in enumerate(zip(prompt_markers_list, list_prefix)):
        

        for layer_name, embedding_batch in full_embeddings.items():
            
            # Create the save directory if it doesn't exist
                # Create a safe filename from the layer name
            safe_name = layer_name.replace('.', '_').replace('/', '_')
            base_dir = os.path.join(save_dir, safe_name)
            os.makedirs(base_dir, exist_ok=True)
            if prefix:
                safe_name = f"{prefix}_{safe_name}"

            if isinstance(embedding_batch, tuple):
                # Handle tuple by taking first element
                embedding_batch = embedding_batch[0]

            # Save single tensor
            if not torch.is_tensor(embedding_batch):
                print('not single tensor', layer_name)
                
                embedding_batch = torch.tensor(embedding_batch)
            #print('embedding_batch.shape', layer_name, embedding_batch.shape)
            #log_to_file(counter,layer_name, embedding.shape)
            if 'language' in layer_name:
                embedding = embedding_batch[0]
                
                #take last 7. TOTAL 7
                embedding_last7 = embedding[-7:,:]
                if not simple_extraction:
                    #print('embedding_last7.dtype', layer_name, embedding_last7.dtype)
                    #take first 1. TOTAL 1
                    embedding_first1 = embedding[0,:].unsqueeze(0)
                    #print('embedding_first1.dtype', layer_name, embedding_first1.dtype)
                    #print('first 1 embeddings', embedding_first1.shape)

                    #if there is a pre, get it, 1 avg and last 7. TOTAL 8
                    #how many pre can we fill
                    avail_pre = max(prompt_markers['pre_end_index'] - prompt_markers['pre_start_index'] +1,0)
                    #print('avail_pre', avail_pre)
                    if(avail_pre > 0):
                        #get average of all tokens
                        embedding_pre = embedding[prompt_markers['pre_start_index']:prompt_markers['pre_end_index']+1,:]
                        embedding_pre_avg = torch.mean(embedding_pre, dim=0).unsqueeze(0)
                    else:
                        #put in empty tensors
                        embedding_pre_avg = torch.zeros(1, embedding.shape[1], device=embedding.device, dtype=embedding.dtype)
                    #take last 7. TOTAL 7
                    num_real = min(7, avail_pre)
                    num_0 = 7 - num_real
                    embeddings_last7_pre = None
                    if(avail_pre > 0):
                        embeddings_last7_pre =embedding[prompt_markers['pre_end_index'] - num_real+1:prompt_markers['pre_end_index']+1,:]
                        #print('real embeddings_last7_pre', embeddings_last7_pre.shape)
                    if num_0 > 0:
                        if embeddings_last7_pre is None:
                            embeddings_last7_pre = torch.zeros(num_0, embedding.shape[1], device=embedding.device, dtype=embedding.dtype)
                            #print('empty embeddings_last7_pre', embeddings_last7_pre.shape)
                        else:
                            embeddings_last7_pre = torch.cat([embeddings_last7_pre, torch.zeros(num_0, embedding.shape[1], device=embedding.device, dtype=embedding.dtype)], dim=0)
                            #print('real + empty embeddings_last7_pre', embeddings_last7_pre.shape)
                    embedding_pre = torch.cat([embedding_pre_avg, embeddings_last7_pre], dim=0)
                    #print('embedding_pre.dtype', layer_name, embedding_pre.dtype)
                    #print('embedding_pre', embedding_pre.shape)

                    #if there is a post, get it, 1 avg and last 7. TOTAL 8
                    #how many post can we fill
                    #print('post_start_index', prompt_markers['post_start_index'], counter)
                    #print('post_end_index', prompt_markers['post_end_index'])
                    avail_post = max(prompt_markers['post_end_index'] - prompt_markers['post_start_index'] +1,0)
                    #print('avail_post', avail_post)
                    if(avail_post > 0):
                        #get average of all tokens
                        embedding_post = embedding[prompt_markers['post_start_index']:prompt_markers['post_end_index']+1,:]
                        embedding_post_avg = torch.mean(embedding_post, dim=0).unsqueeze(0)
                    else:
                        #put in empty tensors
                        embedding_post_avg = torch.zeros(1, embedding.shape[1], device=embedding.device, dtype=embedding.dtype)
                    #take last 7. TOTAL 7
                    num_real = min(7, avail_post)
                    #print('num_real', num_real)
                    num_0 = 7 - num_real
                    #print('num_0', num_0)
                    embeddings_last7_post = None
                    if(avail_post > 0):
                        embeddings_last7_post =embedding[prompt_markers['post_end_index'] - num_real+1:prompt_markers['post_end_index']+1,:]
                        #print('real embeddings_last7_post', embeddings_last7_post.shape)
                    if num_0 > 0:
                        if embeddings_last7_post is None:
                            embeddings_last7_post = torch.zeros(num_0, embedding.shape[1], device=embedding.device, dtype=embedding.dtype)
                            #print('empty embeddings_last7_post', embeddings_last7_post.shape)
                        else:
                            embeddings_last7_post = torch.cat([embeddings_last7_post, torch.zeros(num_0, embedding.shape[1], device=embedding.device, dtype=embedding.dtype)], dim=0)
                            #print('real + empty embeddings_last7_post', embeddings_last7_post.shape)
                    embedding_post = torch.cat([embedding_post_avg, embeddings_last7_post], dim=0)
                    #print('embedding_post.dtype', layer_name, embedding_post.dtype)
                    #print('embedding_post', embedding_post.shape)
                    
                    
                    #save average of individual image so 8 tensors + last 1 at end of image that is all combine. TOTAL 9 
                    first_avg = True
                    for (start, end) in prompt_markers['img_index_values']:
                        image_embedding = embedding[start+1:end,:]
                        #print('image_embedding', image_embedding.shape)
                        if first_avg:
                            image_embedding_avg = torch.mean(image_embedding, dim=0).unsqueeze(0)
                            first_avg = False
                        else:
                            image_embedding_avg = torch.cat([image_embedding_avg, torch.mean(image_embedding, dim=0).unsqueeze(0)], dim=0)
                        #print('image_embedding_avg', image_embedding_avg.shape)
                        #exit()
                    #print('image_embedding_avg', image_embedding_avg.shape)
                    #add last token for end of last chat
                    all_img_embeddings = torch.cat([image_embedding_avg, embedding[prompt_markers['img_index_values'][-1][-1],:].unsqueeze(0)], dim=0)
                    #print('all_img_embeddings.dtype', layer_name, all_img_embeddings.dtype)
                    #print('all_img_embeddings', all_img_embeddings.shape)
                    #
                    # print('embedding_last7', embedding_last7.shape)
                    # print('embedding_first1', embedding_first1.shape)
                    # print('embedding_pre', embedding_pre.shape)
                    # print('embedding_post', embedding_post.shape)
                    # print('all_img_embeddings', all_img_embeddings.shape)
                    embedding = torch.cat([embedding_last7, embedding_first1, embedding_pre, embedding_post, all_img_embeddings], dim=0)
                else:
                    #print('embedding.shape', embedding.shape)
                    embedding = embedding_last7
                assert embedding.dtype == torch.bfloat16, f"embedding.dtype {embedding.dtype} != torch.bfloat16"
            if 'vision' in layer_name:
                img_start = i*8
                img_end = img_start + 8
                # print('vision', embedding_batch.shape, len(prompt_markers_list))
                embedding_cls = embedding_batch[img_start:img_end,0,:]
                # print('embedding_cls', embedding_cls.shape)
                embedding_non_cls = embedding_batch[img_start:img_end,1:,:]
                # print('embedding_non_cls', embedding_non_cls.shape)
                avg_embedding_non_cls = torch.mean(embedding_non_cls, dim=1)
                # print('avg_embedding_non_cls', avg_embedding_non_cls.shape)
                embedding = torch.cat([embedding_cls, avg_embedding_non_cls], dim=0)
                # print('embedding vision', embedding.shape)
            with gzip.open(os.path.join(base_dir, safe_name + file_ext), 'wb') as f:
                pickle.dump(embedding.cpu(), f)

            # with h5py.File(os.path.join(save_dir, safe_name + file_ext), 'w') as f:
            #     f.create_dataset('data', data=embedding.numpy())#, compression="gzip")
            metadata[layer_name] = {
                'type': 'tensor',
                'shape': list(embedding.shape) if hasattr(embedding, 'shape') else None
            }
        
        metadata['text'] = text
        # Save metadata
        with open(os.path.join(save_dir, 'metadata',f"{prefix}_metadata.json" if prefix else "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)

def load_embeddings(save_dir, prefix="", use_numpy=False):
    """
    Load embeddings from files.
    
    Args:
        save_dir: Directory containing the saved embeddings
        prefix: Optional prefix used when saving
        use_numpy: If True, load as numpy arrays. If False, load as PyTorch tensors
    
    Returns:
        Dictionary of layer names to embeddings
    """
    # Load metadata
    metadata_file = os.path.join(save_dir, f"{prefix}_metadata.json" if prefix else "metadata.json")
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    embeddings = {}
    for layer_name, info in metadata.items():
        safe_name = layer_name.replace('.', '_').replace('/', '_')
        if prefix:
            safe_name = f"{prefix}_{safe_name}"
            
        # Load the embedding
        if use_numpy:
            emb = np.load(os.path.join(save_dir, f"{safe_name}.npy"))
        else:
            emb = torch.load(os.path.join(save_dir, f"{safe_name}.pt"))
            
        # If it was originally a tuple, wrap it back in a tuple
        if info['type'] == 'tuple_first':
            embeddings[layer_name] = (emb,)
        else:
            embeddings[layer_name] = emb
    
    return embeddings

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
                #print('output.last_hidden_state ',name, output.last_hidden_state.shape)
                layer_outputs[name] = output.last_hidden_state.detach()
            elif isinstance(output, tuple):
                #print('output tuple', name)
                layer_outputs[name] = tuple(x.detach() if torch.is_tensor(x) else x for x in output)
            elif torch.is_tensor(output):
                #print('output tensor', name, output.shape)
                layer_outputs[name] = output.detach()
            else:
                #print('output else', name)
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

# # Define special tokens
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
def get_params_for_forward_no_pix(model,tokenizer, text_prompt, counter):
    template = get_conv_template(model.template)
    template.append_message(template.roles[0], text_prompt)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    model_inputs = tokenizer(query, return_tensors='pt', return_offsets_mapping=True)
    input_ids = model_inputs['input_ids'].to(model.device)

    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    if model.img_context_token_id is None:
        model.img_context_token_id = img_context_token_id

    return input_ids

def get_params_for_forward(model,tokenizer, pixel_values, text_prompt, counter):


    #phase 1
    model_inputs = tokenizer(text_prompt, return_tensors='pt')
    input_ids = model_inputs['input_ids'].to(model.device)
    #phase 2
    template = get_conv_template(model.template)
    if '<image>' not in text_prompt:
        text_prompt = '<image>\n' + text_prompt
    template.append_message(template.roles[0], text_prompt)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()
    # Phase 3 add IMG_CONTEXT_TOKEN
    num_patches = 1
    image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token  + IMG_END_TOKEN
    query = query.replace('<image>', image_tokens, pixel_values.shape[0])
    # # Tokenize the query
    model_inputs = tokenizer(query, return_tensors='pt', return_offsets_mapping=True)
    input_ids = model_inputs['input_ids'].to(model.device)
    offsets = model_inputs.offset_mapping
    #set img_context_token_id
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    img_start_token_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
    img_end_token_id = tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
    if model.img_context_token_id is None:
        model.img_context_token_id = img_context_token_id


    image_flags = torch.ones(pixel_values.shape[0], dtype=torch.long, device=model.device)

    def get_img_index_values(input_ids, img_start_token_id, img_end_token_id):
    #populate key index values from inputs
        start_positions = torch.where(input_ids[0] == img_start_token_id)[0]
        end_positions = torch.where(input_ids[0] == img_end_token_id)[0]
        
        # Convert to list and pair them up
        img_index_values = list(zip(start_positions.tolist(), end_positions.tolist()))
        return img_index_values
        # img_index_values = []
        # start_index = None
        # for i in range(input_ids.shape[1]):
        #     if input_ids[0, i] == img_start_token_id:
        #         start_index = i
        #     elif input_ids[0, i] == img_end_token_id:
        #         img_index_values.append((start_index, i))
        #         start_index = None
        # return img_index_values

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
        chat_start_token_id = tokenizer.convert_tokens_to_ids(chat_start_word)
        chat_end_token_id = tokenizer.convert_tokens_to_ids(chat_end_word)
        # Use vectorized operations to find all chat start and end positions
        chat_start_positions = torch.where(input_ids[0] == chat_start_token_id)[0]
        chat_end_positions = torch.where(input_ids[0] == chat_end_token_id)[0]
        
        # Convert to lists for indexing
        chat_start_index_values = chat_start_positions.tolist()
        chat_end_index_values = chat_end_positions.tolist()
        assert len(chat_end_index_values) == 2, f"len(chat_end_index_values) {len(chat_end_index_values)} != 2"
        assert len(chat_start_index_values) == 3, f"len(chat_start_index_values) {len(chat_start_index_values)} != 3"
        last_chat_end_index = chat_end_index_values[1]
        #4 tokens after start of second chat
        pre_start_index = chat_start_index_values[1] + 4
        #4 tokens before the start of first image
        pre_end_index = img_index_values[0][0] - 5
        #4 chars after the last image end
        post_start_index = img_index_values[len(img_index_values)-1][1] + 5
        #1 char before the last chat end
        post_end_index = chat_end_index_values[1] - 1
        return pre_start_index, pre_end_index, post_start_index, post_end_index, last_chat_end_index
    
    img_index_values = get_img_index_values(input_ids, img_start_token_id, img_end_token_id)
    check_img_index_values(img_index_values)
    pre_start_index, pre_end_index, post_start_index, post_end_index, last_chat_end_index = get_pre_post_index_values(input_ids, tokenizer, img_index_values)

    prompt_markers ={
        'pre_start_index': pre_start_index,
        'pre_end_index': pre_end_index,
        'post_start_index': post_start_index,
        'post_end_index': post_end_index,
        'last_chat_end_index': last_chat_end_index,
        'img_index_values': img_index_values,
    }
    # utils.log_to_file(f"counter: {counter}")
    # utils.print_input_tokens_with_offsets(query, offsets, input_ids, pre_start=pre_start_index, pre_end=pre_end_index, post_start=post_start_index, post_end=post_end_index, last_chat_end=last_chat_end_index)

    return input_ids, image_flags, prompt_markers

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
    
    # Convert all to tensors
    input_ids_tensors = input_ids_list
    # for ids in input_ids_list:
    #     if isinstance(ids, list):
    #         ids = torch.tensor(ids)
    #     input_ids_tensors.append(ids)
    
    # Find max length
    if max_length is None:
        max_length = max(len(ids) for ids in input_ids_tensors)
    # print("max_length", max_length)
    # Pad sequences and create attention masks
    batch_input_ids = []
    batch_attention_mask = []
    
    for ids in input_ids_tensors:
        current_length = len(ids)
        
        if current_length < max_length:
            #print('***doing padding***', current_length, max_length)
            #utils.log_to_file(f"current_length: {current_length}, max_length: {max_length}")
            # Pad if shorter than max_length
            padding_length = max_length - current_length
            paddings = torch.full((padding_length,), pad_token_id, dtype=ids.dtype, device=device)
            # print("paddings", paddings.shape)
            # print("ids", ids.shape)
            ids_padded = torch.cat([paddings, ids])
            attention = torch.cat([torch.zeros(padding_length, dtype=torch.long, device=device),
                                   torch.ones(current_length, dtype=torch.long, device=device)])
            ids = ids_padded
        else:
            attention = torch.ones(max_length, dtype=torch.long, device=device)
        batch_input_ids.append(ids)
        batch_attention_mask.append(attention)
    
    return torch.stack(batch_input_ids), torch.stack(batch_attention_mask)

def get_embeddings_with_existing_hooks_forward(model, tokenizer, list_pixel_values, list_text_prompt, layer_outputs, counter, skip_pix=False):
    """
    Get embeddings using existing hooks
    
    Args:
        model: The model
        tokenizer: The tokenizer
        pixel_values: The image tensor
        text_prompt: The text prompt
        layer_outputs: Dictionary to store layer outputs
        
    Returns:
        Dictionary of layer outputs
    """
    # Clear previous outputs
    layer_outputs.clear()
    #forward method implementation
    assert len(list_pixel_values) == len(list_text_prompt)
    final_input_ids = []
    final_image_flags = None
    final_pixel_values = None
    prompt_markers_list = []
    if not skip_pix:
        for pixel_values, text_prompt in zip(list_pixel_values, list_text_prompt):
            input_ids, image_flags, prompt_markers = get_params_for_forward(model, tokenizer, pixel_values, text_prompt, counter)
            final_input_ids.append(input_ids.squeeze(0))
            prompt_markers_list.append(prompt_markers)
            if final_image_flags is None:
                final_image_flags = image_flags
            else:
                final_image_flags = torch.cat((final_image_flags, image_flags), dim=0)
            if final_pixel_values is None:
                final_pixel_values = pixel_values
            else:
                final_pixel_values = torch.cat((final_pixel_values, pixel_values), dim=0)


        #print('final_input_ids.shape', final_input_ids[0].shape)
        final_input_ids, final_attention_mask = generate_attention_mask_batch(final_input_ids, tokenizer.pad_token_id, model.device)
        final_input_ids = final_input_ids[0].unsqueeze(0)
        assert len(prompt_markers_list) == len(list_text_prompt)
        #print('final_input_ids.shape', final_input_ids.shape)
    else:
        final_pixel_values = None
        final_image_flags = None
        final_input_ids = get_params_for_forward_no_pix(model, tokenizer, list_text_prompt[0], counter)
        prompt_markers_list.append(True)

    with torch.no_grad():
        model(
            pixel_values=final_pixel_values,
            input_ids=final_input_ids,
            image_flags=final_image_flags,
            return_dict=False,
            output_hidden_states  = False,
            #attention_mask=final_attention_mask
        )
    return dict(layer_outputs), prompt_markers_list


def get_embeddings_with_existing_hooks(model, tokenizer, list_pixel_values, list_text_prompt, layer_outputs, counter):
    """
    Get embeddings using existing hooks
    
    Args:
        model: The model
        tokenizer: The tokenizer
        pixel_values: The image tensor
        text_prompt: The text prompt
        layer_outputs: Dictionary to store layer outputs
        
    Returns:
        Dictionary of layer outputs
    """
    # Clear previous outputs
    layer_outputs.clear()
    #forward method implementation
    assert len(list_pixel_values) == len(list_text_prompt)
    final_input_ids = []
    final_image_flags = None
    final_pixel_values = None
    prompt_markers_list = []
    for pixel_values, text_prompt in zip(list_pixel_values, list_text_prompt):
        prompt_markers_list.append(True)
        if final_pixel_values is None:
            final_pixel_values = pixel_values
        else:
            final_pixel_values = torch.cat((final_pixel_values, pixel_values), dim=0)
        
    
    assert len(prompt_markers_list) == len(list_text_prompt)
    #print('final_input_ids.shape', final_input_ids.shape)

    #chat method implementation
    generation_config = dict(
        max_new_tokens=1,
        pad_token_id=tokenizer.pad_token_id  # Explicitly set to avoid warning
    )
    pixel_values = final_pixel_values
    num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
    if len(list_text_prompt) > 1:   
        #print('batch prompt')
        response = model.batch_chat(
            tokenizer, 
            pixel_values, 
            list_text_prompt, 
            generation_config,
            num_patches_list=num_patches_list,
            history=None, 
            return_history=False,
            verbose=False,
            #output_hidden_states=True
        )
    else: 
        #print('single prompt')
        text_prompt = list_text_prompt[0]
        #print('text_prompt', text_prompt)
        response = model.chat(
            tokenizer, 
            pixel_values, 
            text_prompt, 
            generation_config,
            history=None, 
            return_history=False,
            verbose=False,
            #output_hidden_states=True
        )

    #utils.log_to_file(counter, ':', response)
    return dict(layer_outputs), prompt_markers_list

def process_all_files_for_embedding_extraction():
    root_data_dir = utils.get_data_root_dir()
# As an exemple, extract visual features for season 1, episode 1 of Friends
    #episode_path = root_data_dir + "algonauts_2025.competitors/stimuli/movies/friends/s1/friends_s01e01a.mkv"
    # Collecting the paths to all the movie stimuli
    stimuli_prefix = utils.get_stimuli_prefix()
    #if stimuli_prefix is None
    file_in_filter = ''
    exclude_list = []#['friends_s03e05b', 'friends_s03e06a']
    files = glob(f"{root_data_dir}/algonauts_2025.competitors/stimuli/movies/friends/{stimuli_prefix}/*.mkv")

    if file_in_filter:
        files = [f for f in files if file_in_filter in f]
    files.sort()

    stimuli = {f.split("/")[-1].split(".")[0]: f for f in files}
    print(len(stimuli), list(stimuli)[:3], list(stimuli)[-3:])

    # Duration of each movie chunk, aligned with the fMRI TR of 1.49 seconds
    tr = 1.49

    

    # Saving directories
    save_dir_temp = utils.get_tmp_dir()
    hf_path = utils.get_mvl_model() #"OpenGVLab/InternVL3-1B-Pretrained"
    device_map = split_model(hf_path)
    simple_extraction = utils.get_mvl_simple_extraction()
    # For the second model loading instance
    # Configure quantization
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )
    if utils.isMockMode():
        model = None
    else:
        model = AutoModel.from_pretrained(
            hf_path,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map).eval()
    tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True, use_fast=True)
    # Add this line to set pad_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if not simple_extraction:
        custom_layers = [
                    'vision_model.encoder.layers.2',
                    'vision_model.encoder.layers.4',
                    'vision_model.encoder.layers.12',
                    'vision_model.encoder.layers.22',
                    'vision_model.encoder.layers.23',
                    'vision_model',                     # Vision encoder
                    'language_model.model.layers.4',    # First layer
                    'language_model.model.layers.12',    # First layer
                    'language_model.model.layers.20',    # First layer
                    'language_model.model.layers.21',    # Middle layer
                    'language_model.model.layers.22',   # Later layer
                    'language_model.model.layers.23',   # Later layer
                    'language_model.model.norm'         # Final normalization
                ]
    else:
        custom_layers = [
            'language_model.model.norm'         # Final normalization
        ]

    # Set up hooks once before processing all files
    if utils.isMockMode():
        hooks, layer_outputs, hook_storage = None, None, None
        use_progress_bar = False
    else:
        hooks, layer_outputs, hook_storage = create_layer_hooks(model, custom_layers)
        use_progress_bar = True
    
    try:
        # iterate across all the stimuli movie files
        iterator = tqdm(enumerate(stimuli.items()), total=len(list(stimuli)), disable= not use_progress_bar)
        for i, (stim_id, stim_path) in iterator:
            print(f"Extracting visual features for {stim_id}", stim_path)
            if stim_id in exclude_list:
                continue
            text_dataset = get_transcript_dataSet(stim_id)
            transcript_file = stim_path.replace('.mkv', '.tsv').replace('movies', 'transcripts')
            # Pass layer_outputs to the extraction function
            extract_vlm_embeddings(stim_id, text_dataset, model, tokenizer, 
                                 layer_outputs, use_progress_bar)
    finally:
        # Clean up hooks after all processing is done
        if not utils.isMockMode():
            for h in hooks:
                h.remove()

def save_pixel_values(pixel_values, save_dir, prefix):
    """
    Save pixel_values tensor to a compressed file.
    
    Args:
        pixel_values (torch.Tensor): The pixel values tensor to save
        save_dir (str): Directory to save the file in
        prefix (str): Prefix for the filename (e.g. video chunk ID)
    """
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{prefix}.pt.gz")
    
    # Convert to CPU if on GPU
    if torch.is_tensor(pixel_values) and pixel_values.is_cuda:
        pixel_values = pixel_values.cpu()
        
    with gzip.open(file_path, 'wb') as f:
        pickle.dump(pixel_values, f)


def get_num_chunks(episode_id):
    season = 's1'
    if 's02' in episode_id:
        season = 's2'
    elif 's03' in episode_id:
        season = 's3'
    elif 's04' in episode_id:
        season = 's4'
    elif 's05' in episode_id:
        season = 's5'
    elif 's06' in episode_id:
        season = 's6'
    season_folder = os.path.join(utils.get_output_dir(), 'video_chunks', season)
    files = glob(f"{season_folder}/{episode_id}_*.mp4")
    return len(files), season_folder

def extract_vlm_embeddings(episode_id, text_dataset, model, tokenizer, 
                          layer_outputs, use_progress_bar):
    num_chunks, season_folder = get_num_chunks(episode_id)
    # print('num_chunks', num_chunks)
    # print('season_folder', season_folder)
    # Empty features list
    extracted_features = []
    counter = 0
    n_used_words = 1000
    batch_size = utils.get_mvl_batch_size()

    len_trans_dataset = len(text_dataset)
    assert num_chunks -1 <= len_trans_dataset <= num_chunks, f"len(trans_dataset) != num_chunks {len_trans_dataset} != {num_chunks}"
    # if len_trans_dataset != num_chunks:
    #     print('len(trans_dataset) != num_chunks', len_trans_dataset, num_chunks)
    #assert len(trans_dataset) == len(start_times), f"len(dataset) = {len(trans_dataset)} != len(start_times) = {len(start_times)}"	
    # Loop over chunks
    # Helper function to create batches
    def create_batches(data_length, batch_size):
        for i in range(0, data_length, batch_size):
            yield range(i, min(i + batch_size, data_length))

    # batch_generator = create_batches(len_trans_dataset, batch_size)
    # for batch_indices in batch_generator:
    #     print(f"Processing batch {batch_indices}")
        # Process each batch here
    skip_pix = utils.get_mvl_skip_pix()
    with tqdm(total=len_trans_dataset, desc="Extracting ..", disable= not use_progress_bar) as pbar:
        for batch_indices in create_batches(len_trans_dataset, batch_size):
            #print(f"Processing batch {batch_indices}")
            embeddings_dir = os.path.join(utils.get_output_dir(), utils.get_embeddings_dir())
            pixel_values_list = []
            question_for_embeddings_list = []
            embeddings_prefix_list = []
            for counter in batch_indices:
                # if counter > 100:
                #     break
                embeddings_prefix = f"{episode_id}_tr_{counter}"
                meta_file = os.path.join(embeddings_dir, 'metadata', f"{embeddings_prefix}_metadata.json")
                if not os.path.exists(meta_file):
                    chunk_path = os.path.join(season_folder, f'{episode_id}_tr_{counter}.mp4')
                    #log_to_file(counter,'chunk_path', chunk_path)
                    # Load the frames from the chunked movie clip
                    trans_index = counter
                    if skip_pix:
                        pixel_values = torch.randn(8, 3, 448, 448, dtype=torch.bfloat16)
                    else:
                        pixel_values, num_patches_list = utils_video.load_video(chunk_path, num_segments=8, max_num=1)
                        
                    pixel_values = pixel_values.to(torch.bfloat16).cuda()
                    textData = text_dataset[trans_index]

                    pre_text = textData['fancy_pre']
                    post_text = textData['fancy_post']

                    if skip_pix:
                        if pre_text:
                            video_prefix = pre_text
                        else:
                            video_prefix = ''
                    else:
                        if pre_text:
                            video_prefix = pre_text + "\n" + ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
                        else:
                            video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
                    
                    
                    if post_text:
                        question_for_embeddings = video_prefix + "\n" + post_text
                    else:
                        question_for_embeddings = video_prefix + "\n"

                    #utils.log_to_file(counter,':', question_for_embeddings)
                    pixel_values_list.append(pixel_values)
                    question_for_embeddings_list.append(question_for_embeddings)
                    embeddings_prefix_list.append(embeddings_prefix)

            if len(question_for_embeddings_list) > 0 and not utils.isMockMode():
                extracted_features, prompt_markers_list = get_embeddings_with_existing_hooks_forward(
                    model, 
                    tokenizer, 
                    pixel_values_list, 
                    question_for_embeddings_list,
                    layer_outputs,
                    counter,
                    skip_pix=skip_pix
                )

                
                save_embeddings(extracted_features, prompt_markers_list, embeddings_dir, text=text_dataset[counter], 
                        list_prefix=embeddings_prefix_list, counter=counter)
            pbar.update(len(batch_indices)) if use_progress_bar else None
def get_transcript_dataSet(stim_id):
    root_data_dir = utils.get_data_root_dir()
    transcript_data, trans_info_list, total_tr_len = load_all_tsv_for_one_episode(stim_id[:-1], isEnhanced=True)
    tr_start = 0
    tr_length =0
    for tr_info in trans_info_list:
        if tr_info['trans_id'] == stim_id:
            tr_length = tr_info['len']
            break
        else:
            tr_start += tr_info['len']

    dialogue_file = os.path.join(root_data_dir, 'algonauts_2025.competitors','stimuli', 'transcripts', 'friends', 'full', f'{stim_id[:-1]}.txt')
    dialogues = get_scene_dialogue(dialogue_file)
    trans_dataset = SentenceDataset_v2(transcript_data, dialogues, tr_start, tr_length, always_post_speaker=True, exclude_post_dialogue_separator=True)
    return trans_dataset

def wrap_text(text, max_length):
    words = text.split()
    wrapped_text = []
    current_line = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 > max_length:
            # If current line is not empty, join it and add to wrapped_text
            if current_line:
                wrapped_text.append(' '.join(current_line))
                current_line = []
                current_length = 0
            
            # If the word itself is longer than max_length, split it
            while len(word) > max_length:
                wrapped_text.append(word[:max_length])
                word = word[max_length:]
            
            # Add the remaining part of the word (or the whole word if it fits)
            if word:
                current_line = [word]
                current_length = len(word)
        else:
            # Add word to current line
            if current_line:
                current_length += 1  # For the space
            current_line.append(word)
            current_length += len(word)
    
    # Add the last line if it's not empty
    if current_line:
        wrapped_text.append(' '.join(current_line))
    
    return '\n'.join(wrapped_text)

def test_dataset(stim_id):
    #stim_id = 'friends_s01e23a'
    trans_dataset = get_transcript_dataSet(stim_id)
    data = []
    maxcolwidths=[30, 30, 40, 35, 20, 5, 5]
    for i in range(len(trans_dataset)):
        textData= trans_dataset[i]
        pre_text = textData['fancy_pre']
        post_text = textData['fancy_post']
        if pre_text:
            video_prefix = pre_text + "\n" + ''.join([f'Frame{i+1}: <image>\n' for i in range(8)])
        else:
            video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(8)])
        
        if post_text:
            question_for_embeddings = video_prefix + "\n" + post_text
        else:
            question_for_embeddings = video_prefix
        print(question_for_embeddings)
        # print(i, "length", len(trans_dataset))
        # print(response['fancy_pre'])
        # print("\n")
        # print(response['fancy_post'])
        # print('*'*100)
        #
        #

        # Replace None values with empty strings to prevent tabulate error
    #     fancy_pre = wrap_text(response['fancy_pre'], 70) if response['fancy_pre'] is not None else ""
    #     normal_pre = wrap_text(response['normal_pre'], 30) if response['normal_pre'] is not None else ""
    #     fancy_post = wrap_text(response['fancy_post'], 30) if response['fancy_post'] is not None else ""
    #     normal_post = wrap_text(response['normal_post'], 30) if response['normal_post'] is not None else ""
    #     words_tr = response['words_tr'] if response['words_tr'] is not None else ""
    #     word_length = response['word_length'] if response['word_length'] is not None else ""
        
    #     data.append([ fancy_pre, fancy_post, words_tr, i])
    # print(tabulate(data, headers=["Fancy Pre", "Fancy Post", "Transcript", "index"], 
    #                tablefmt="grid"))

#process_all_files_for_extraction()
episode_path = "/home/bagga005/algo/comp_data/algonauts_2025.competitors/stimuli/movies/friends/s3/friends_s03e06a.mkv"
save_dir = "/home/bagga005/algo/comp_data/tmp/vid"
stim_id = "friends_s03e06a"
tr = 1.49
#extract_video_chucks()
#extract_save_video_chunks(episode_path, save_dir, stim_id, tr)
#extract_video_chucks()
process_all_files_for_embedding_extraction()

#test_dataset('friends_s03e06a')
#   exit()

# root_data_dir = utils.get_data_root_dir()
# files = glob(f"{root_data_dir}/stimuli/transcripts/friends/s*/*.tsv")
# files.sort()
# #/home/bagga005/algo/comp_data/algonauts_2025.competitors/stimuli/transcripts/friends/s3/friends_s03e02a.tsv
# stimuli = {f.split("/")[-1].split(".")[0]: f for f in files}
# print(len(stimuli), list(stimuli)[:3], list(stimuli)[-3:])
# for stim_id, stim_path in stimuli.items():
#     print(stim_id)
#     test_dataset(stim_id)

