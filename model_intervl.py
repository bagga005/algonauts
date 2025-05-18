import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig
import requests
import os
import json
from utils import get_output_dir

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    if image_file.startswith('http'):
        image = Image.open(requests.get(image_file, stream=True).raw).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

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
def save_embeddings(embeddings, save_dir, prefix="", use_numpy=False):
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
            
        if use_numpy:
            file_ext = ".npy"
            
            # Convert to numpy array if it's a tensor
            if torch.is_tensor(embedding):
                embedding = embedding.detach().cpu().numpy()
                
            if isinstance(embedding, tuple):
                # Handle tuple of tensors
                tuple_dir = os.path.join(save_dir, safe_name)
                os.makedirs(tuple_dir, exist_ok=True)
                for i, emb in enumerate(embedding):
                    if torch.is_tensor(emb):
                        emb = emb.detach().cpu().numpy()
                    np.save(os.path.join(tuple_dir, f"{i}{file_ext}"), emb)
                metadata[layer_name] = {
                    'type': 'tuple',
                    'length': len(embedding),
                    'shapes': [arr.shape if hasattr(arr, 'shape') else None for arr in embedding]
                }
            else:
                # Save single array
                np.save(os.path.join(save_dir, safe_name + file_ext), embedding)
                metadata[layer_name] = {
                    'type': 'array',
                    'shape': embedding.shape if hasattr(embedding, 'shape') else None
                }
        else:
            file_ext = ".pt"
            
            if isinstance(embedding, tuple):
                # Handle tuple of tensors
                tuple_dir = os.path.join(save_dir, safe_name)
                os.makedirs(tuple_dir, exist_ok=True)
                for i, emb in enumerate(embedding):
                    if not torch.is_tensor(emb):
                        emb = torch.tensor(emb)
                    torch.save(emb, os.path.join(tuple_dir, f"{i}{file_ext}"))
                metadata[layer_name] = {
                    'type': 'tuple',
                    'length': len(embedding),
                    'shapes': [t.shape if hasattr(t, 'shape') else None for t in embedding]
                }
            else:
                # Save single tensor
                if not torch.is_tensor(embedding):
                    embedding = torch.tensor(embedding)
                torch.save(embedding, os.path.join(save_dir, safe_name + file_ext))
                metadata[layer_name] = {
                    'type': 'tensor',
                    'shape': list(embedding.shape) if hasattr(embedding, 'shape') else None
                }
    
    # Save metadata
    with open(os.path.join(save_dir, f"{prefix}_metadata.json" if prefix else "metadata.json"), 'w') as f:
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
            
        if info['type'] in ['tuple']:
            # Load tuple of embeddings
            tuple_dir = os.path.join(save_dir, safe_name)
            embeddings_tuple = []
            for i in range(info['length']):
                if use_numpy:
                    emb = np.load(os.path.join(tuple_dir, f"{i}.npy"))
                else:
                    emb = torch.load(os.path.join(tuple_dir, f"{i}.pt"))
                embeddings_tuple.append(emb)
            embeddings[layer_name] = tuple(embeddings_tuple)
        else:
            # Load single embedding
            if use_numpy:
                embeddings[layer_name] = np.load(os.path.join(save_dir, f"{safe_name}.npy"))
            else:
                embeddings[layer_name] = torch.load(os.path.join(save_dir, f"{safe_name}.pt"))
    
    return embeddings
def get_embeddings_with_hooks(model, tokenizer, pixel_values, text_prompt, layer_names=None):
    """
    Extract embeddings from specific layers using hooks.
    
    Args:
        model: The model to extract embeddings from
        tokenizer: The tokenizer to process text input
        pixel_values: The image input tensor
        text_prompt: The text prompt to use
        layer_names: List of layer names to extract embeddings from.
                     If None, extracts from predetermined layers
                     
    Returns:
        A dictionary mapping layer names to their output embeddings
    """
    # If no specific layers are specified, use these default layers of interest
    if layer_names is None:
        layer_names = [
            'vision_model',
            'mlp1',
            'language_model.model.layers.0',
            'language_model.model.layers.5',
            'language_model.model.layers.10',
            'language_model.model.norm'
        ]
    
    # Store embeddings here
    embeddings = {}
    hooks = []
    
    # Define hook function to capture outputs
    def get_hook_fn(layer_name):
        def hook_fn(module, input, output):
            # Handle different output types
            if hasattr(output, 'last_hidden_state'):
                # Structured output like BaseModelOutputWithPooling
                embeddings[layer_name] = {
                    'last_hidden_state': output.last_hidden_state.detach() if hasattr(output, 'last_hidden_state') else None,
                    'pooler_output': output.pooler_output.detach() if hasattr(output, 'pooler_output') else None
                }
            elif isinstance(output, tuple):
                # Some layers return tuples
                embeddings[layer_name] = tuple(o.detach() if torch.is_tensor(o) else o for o in output)
            elif torch.is_tensor(output):
                # Simple tensor output
                embeddings[layer_name] = output.detach()
            else:
                # Other types - store as is
                embeddings[layer_name] = output
        return hook_fn
    
    # Register hooks for each layer
    for layer_name in layer_names:
        if '.' in layer_name:
            # Handle nested module names
            parts = layer_name.split('.')
            module = model
            for part in parts:
                module = getattr(module, part)
            hooks.append(module.register_forward_hook(get_hook_fn(layer_name)))
        else:
            # Handle top-level modules
            hooks.append(getattr(model, layer_name).register_forward_hook(get_hook_fn(layer_name)))
    
    # Process the text input
    print('Tokenizing input text...')
    inputs = tokenizer(text_prompt, return_tensors="pt")
    input_ids = inputs.input_ids.cuda()
    
    # Prepare the image flags (needed for the model)
    # The chat method likely does this automatically
    image_flags = torch.zeros_like(input_ids)
    
    # Find the image tokens in the prompt and set their flags
    for i, token_id in enumerate(input_ids[0]):
        if tokenizer.decode(token_id) == '<image>':
            image_flags[0, i] = 1
    
    # Add any other required parameters based on model inspection
    print('Running forward pass...')
    
    # Forward pass through the model with all required parameters
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids, 
            pixel_values=pixel_values,
            image_flags=image_flags,  # Add this crucial parameter
            output_hidden_states=True  # Request hidden states to be returned
        )
    
    print('Removing hooks...')
    # Remove all hooks
    for hook in hooks:
        hook.remove()
    
    # If the model's forward method returns hidden_states, add them to our embeddings dict
    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
        for i, hidden_state in enumerate(outputs.hidden_states):
            embeddings[f'hidden_state_{i}'] = hidden_state
    
    return embeddings

def get_layer_by_layer_embeddings(model, tokenizer, pixel_values, text_prompt, layers=None):
    """
    Get embeddings from specific layers of the model
    
    Args:
        model: The InternVL model
        tokenizer: The tokenizer
        pixel_values: The image tensor
        text_prompt: The text prompt to process
        layers: List of layer names to extract from (if None, uses defaults)
        
    Returns:
        Dictionary mapping layer names to their embeddings
    """
    # Default layers to extract if none specified
    if layers is None:
        layers = [
            'vision_model',
            'mlp1',  # Vision-language connector
            'language_model.model.layers.0',
            'language_model.model.layers.2',
            'language_model.model.layers.5', 
            'language_model.model.layers.10',
            'language_model.model.norm'
        ]
    
    # Store layer outputs here
    layer_outputs = {}
    hooks = []
    
    # Create hooks for each layer
    for layer_name in layers:
        # Create a function to capture this specific layer's output
        def get_hook(name):
            def hook(module, input, output):
                # Different handling based on output type
                if hasattr(output, 'last_hidden_state'):
                    # For structured outputs like from vision_model
                    layer_outputs[name] = output.last_hidden_state.detach()
                elif isinstance(output, tuple):
                    # For tuple outputs
                    layer_outputs[name] = tuple(x.detach() if torch.is_tensor(x) else x for x in output)
                elif torch.is_tensor(output):
                    # For tensor outputs (most layers)
                    layer_outputs[name] = output.detach()
                else:
                    # For other types
                    layer_outputs[name] = output
            return hook
        
        # Register the hook on the appropriate module
        if '.' in layer_name:
            # For nested modules like 'language_model.model.layers.5'
            parts = layer_name.split('.')
            module = model
            for part in parts:
                module = getattr(module, part)
            hooks.append(module.register_forward_hook(get_hook(layer_name)))
        else:
            # For top-level modules like 'vision_model'
            hooks.append(getattr(model, layer_name).register_forward_hook(get_hook(layer_name)))
    
    # Run the model through model.chat (which we know works)
    # We only need a minimal output for embedding extraction
    print('about to run model.chat')
    response, _ = model.chat(
        tokenizer, 
        pixel_values, 
        text_prompt, 
        dict(max_new_tokens=1),
        history=None, 
        return_history=True
    )
    print('done running model.chat:', response)
    # Remove all hooks
    for h in hooks:
        h.remove()
    
    # Print summary of layers captured
    print("Embeddings extracted from layers:")
    for layer in layers:
        if layer in layer_outputs:
            if torch.is_tensor(layer_outputs[layer]):
                print(f"  {layer}: tensor shape {layer_outputs[layer].shape}")
            elif isinstance(layer_outputs[layer], tuple):
                shapes = [x.shape if torch.is_tensor(x) else type(x) for x in layer_outputs[layer]]
                print(f"  {layer}: tuple of {shapes}")
            else:
                print(f"  {layer}: {type(layer_outputs[layer])}")
        else:
            print(f"  {layer}: Not captured")
    
    return layer_outputs

# Alternative example showing direct model access
def get_direct_model_outputs(model, tokenizer, pixel_values, text_prompt):
    """
    Directly access the model's forward method rather than using chat
    
    Args:
        model: The model
        tokenizer: The tokenizer
        pixel_values: Image input tensor
        text_prompt: Text prompt
        
    Returns:
        The raw model outputs
    """
    # For debugging
    print(f"pixel_values shape: {pixel_values.shape}")
    
    # Capture the relevant embeddings using hooks
    vision_embeddings = []
    language_embeddings = []
    
    def vision_hook(module, input, output):
        if hasattr(output, 'last_hidden_state'):
            vision_embeddings.append(output.last_hidden_state.detach())
        else:
            vision_embeddings.append(output.detach() if torch.is_tensor(output) else output)
    
    def language_hook(module, input, output):
        language_embeddings.append(output.detach() if torch.is_tensor(output) else output)
    
    # Register hooks BEFORE calling model.chat
    vision_hook_handle = model.vision_model.register_forward_hook(vision_hook)
    language_hook_handle = model.language_model.model.norm.register_forward_hook(language_hook)
    
    print("Hooks registered, about to run inference")
    
    # Process the inputs using model.chat with a reasonable number of tokens
    # for a more complete response (if desired)
    response, history = model.chat(
        tokenizer, 
        pixel_values, 
        text_prompt, 
        dict(max_new_tokens=1),  # Increased to get more meaningful response
        history=None, 
        return_history=True
    )
    
    print("Done with inference, checking hook results")
    print(f"Vision embeddings captured: {len(vision_embeddings) > 0}")
    print(f"Language embeddings captured: {len(language_embeddings) > 0}")
    
    # Remove hooks
    vision_hook_handle.remove()
    language_hook_handle.remove()
    
    # Return captured embeddings
    return {
        "vision_embeddings": vision_embeddings[0] if vision_embeddings else None,
        "language_embeddings": language_embeddings[0] if language_embeddings else None,
        "vision_embeddings_shape": vision_embeddings[0].shape if vision_embeddings and torch.is_tensor(vision_embeddings[0]) else None,
        "language_embeddings_shape": language_embeddings[0].shape if language_embeddings and torch.is_tensor(language_embeddings[0]) else None,
        "response": response,
        "history": history
    }

# If you set `load_in_8bit=True`, you will need two 80GB GPUs.
# If you set `load_in_8bit=False`, you will need at least three 80GB GPUs.
path = "OpenGVLab/InternVL3-1B-Pretrained"
device_map = split_model('OpenGVLab/InternVL3-1B-Pretrained')
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=False,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map=device_map).eval()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

#process Image
# set the max number of tiles in `max_num`
# pixel_values = load_image('https://techcrunch.com/wp-content/uploads/2025/02/GettyImages-2197091379.jpg', max_num=12).to(torch.bfloat16).cuda()
# question_for_embeddings = '<image>\nSam Altman is the CEO of OpenAI. He is known for his ability to deal make and raise near endless money for OpenAI.'
generation_config = dict(max_new_tokens=1024, do_sample=True)

#process video
video_path = 'temp/red-panda.mp4'
pixel_values, num_patches_list = load_video(video_path, num_segments=7, max_num=1)
pixel_values = pixel_values.to(torch.bfloat16).cuda()
video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
question_for_embeddings = video_prefix + 'What is the red panda doing?'
# Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
print('question_for_embeddings:', question_for_embeddings)

# Example of extracting embeddings using hooks

layer_names = [
    'vision_model',  # Vision encoder embeddings
    'language_model.model.layers.0',  # First LLM layer
    'language_model.model.layers.5',  # Middle LLM layer
    'language_model.model.norm'      # Final normalization layer
]
#outputs = get_direct_model_outputs(model, tokenizer, pixel_values, question_for_embeddings)
#print(outputs)
# Example usage:
custom_layers = [
    'vision_model',                     # Vision encoder
    'mlp1',                             # Vision-language connector 
    'language_model.model.layers.0',    # First layer
    'language_model.model.layers.5',    # Middle layer
    'language_model.model.layers.10',   # Later layer
    'language_model.model.norm'         # Final normalization
]

embeddings = get_layer_by_layer_embeddings(
    model, 
    tokenizer, 
    pixel_values, 
    question_for_embeddings,
    custom_layers
)

embeddings_dir = os.path.join(get_output_dir(), 'embeddings')
save_embeddings(embeddings, embeddings_dir, prefix="image_embeddings", use_numpy=False)

# Now you can work with the embeddings:
for layer_name, embedding in embeddings.items():
    if not torch.is_tensor(embedding):
        continue
        
    # Example: Compute statistics
    mean_val = embedding.mean().item()
    std_val = embedding.std().item()
    print(f"{layer_name} statistics: Mean={mean_val:.4f}, Std={std_val:.4f}")
    
    # Example: Get the first token's embedding from each layer
    if embedding.dim() >= 2:
        first_token = embedding[:, 0]
        print(f"  First token shape: {first_token.shape}")
    
    # Example: Compute cosine similarity between first and last token
    if embedding.dim() >= 2 and embedding.size(1) > 1:
        from torch.nn.functional import cosine_similarity
        first = embedding[:, 0]
        last = embedding[:, -1]
        sim = cosine_similarity(first.flatten(), last.flatten(), dim=0)
        print(f"  First-Last token similarity: {sim.item():.4f}")

exit()
embeddings = get_embeddings_with_hooks(model, tokenizer, pixel_values, question_for_embeddings, layer_names)

# Print embedding shapes to verify
for layer_name, emb in embeddings.items():
    if isinstance(emb, tuple):  # Some layers might return tuples
        print(f"{layer_name} output shape: {[e.shape for e in emb]}")
    else:
        print(f"{layer_name} output shape: {emb.shape}")

# single-image multi-round conversation (单图多轮对话)
# question = '<image>\nPlease describe the image in detail.'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# question = 'Please write a poem according to the image.'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# # multi-image multi-round conversation, combined images (多图多轮对话，拼接图像)
# pixel_values1 = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
# pixel_values2 = load_image('./examples/image2.jpg', max_num=12).to(torch.bfloat16).cuda()
# pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

# question = '<image>\nDescribe the two images in detail.'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config,
#                                history=None, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# question = 'What are the similarities and differences between these two images.'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config,
#                                history=history, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# # multi-image multi-round conversation, separate images (多图多轮对话，独立图像)
# pixel_values1 = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
# pixel_values2 = load_image('./examples/image2.jpg', max_num=12).to(torch.bfloat16).cuda()
# pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
# num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

# question = 'Image-1: <image>\nImage-2: <image>\nDescribe the two images in detail.'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config,
#                                num_patches_list=num_patches_list,
#                                history=None, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# question = 'What are the similarities and differences between these two images.'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config,
#                                num_patches_list=num_patches_list,
#                                history=history, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# # batch inference, single image per sample (单图批处理)
# pixel_values1 = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
# pixel_values2 = load_image('./examples/image2.jpg', max_num=12).to(torch.bfloat16).cuda()
# num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]
# pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

# questions = ['<image>\nDescribe the image in detail.'] * len(num_patches_list)
# responses = model.batch_chat(tokenizer, pixel_values,
#                              num_patches_list=num_patches_list,
#                              questions=questions,
#                              generation_config=generation_config)
# for question, response in zip(questions, responses):
#     print(f'User: {question}\nAssistant: {response}')

# video multi-round conversation (视频多轮对话)
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices


# Demonstrate a more detailed example for extracting embeddings from an image
def extract_and_analyze_embeddings(model, tokenizer, image_path, question, layer_names=None):
    """
    Extract embeddings from an image and analyze them
    
    Args:
        model: The model
        tokenizer: The tokenizer
        image_path: Path to the image file
        question: Question to ask about the image
        layer_names: Layers to extract embeddings from
    
    Returns:
        Dictionary of layer names to embeddings
    """
    # Load and process the image
    pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
    
    # Get embeddings from hooks
    embeddings = get_embeddings_with_hooks(
        model, tokenizer, pixel_values, question, layer_names
    )
    
    return embeddings

# Uncomment the code below to extract embeddings from a specific image with a specific question
# Example usage:
# custom_layers = [
#     'vision_model',
#     'language_model.model.layers.0',
#     'language_model.model.layers.5',
#     'language_model.model.layers.10',
#     'language_model.model.norm'
# ]
# embeddings = extract_and_analyze_embeddings(
#     model,
#     tokenizer,
#     './examples/image1.jpg',
#     '<image>\nWhat objects can you see in this image?',
#     custom_layers
# )
# 
# # Process and use the embeddings as needed
# for layer_name, emb in embeddings.items():
#     # Print shape information
#     if isinstance(emb, tuple):
#         print(f"{layer_name} output shape: {[e.shape for e in emb]}")
#     else:
#         print(f"{layer_name} output shape: {emb.shape}")
#     
#     # Example: You can compute statistics on the embeddings
#     if not isinstance(emb, tuple):
#         print(f"  Mean: {emb.mean().item():.4f}, Std: {emb.std().item():.4f}")
#         print(f"  Min: {emb.min().item():.4f}, Max: {emb.max().item():.4f}")