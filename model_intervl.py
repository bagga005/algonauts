import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from moviepy.editor import VideoFileClip
from torchvision.models.feature_extraction import create_feature_extractor
from pytorchvideo.transforms import Normalize, UniformTemporalSubsample, ShortSideScale
import requests
import os
import json
import utils
from glob import glob
from tqdm import tqdm
import pandas as pd
from model_intervl3 import SentenceDataset
import gzip
import pickle

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
def save_embeddings(embeddings, save_dir, text="", prefix=""):
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
        if 'language' in layer_name:
            #print('language', embedding.shape)
            embedding = embedding.squeeze(0)
            embedding = embedding[-10:,:]
            #print('language', embedding.shape)
        if 'vision' in layer_name:
            #print('vision', embedding.shape)
            embedding = embedding[:,0,:]
            #print('vision', embedding.shape)
        with gzip.open(os.path.join(save_dir, safe_name + file_ext), 'wb') as f:
            pickle.dump(embedding.cpu(), f)
        # with h5py.File(os.path.join(save_dir, safe_name + file_ext), 'w') as f:
        #     f.create_dataset('data', data=embedding.numpy())#, compression="gzip")
        metadata[layer_name] = {
            'type': 'tensor',
            'shape': list(embedding.shape) if hasattr(embedding, 'shape') else None
        }
    
    metadata['text'] = text
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

def get_embeddings_with_existing_hooks(model, tokenizer, pixel_values, text_prompt, layer_outputs):
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
    
    # Run the model
    response, _ = model.chat(
        tokenizer, 
        pixel_values, 
        text_prompt, 
        dict(max_new_tokens=1),
        history=None, 
        return_history=True
    )
    
    return dict(layer_outputs)  # Return a copy of the outputs

def process_all_files_for_extraction():
    root_data_dir = utils.get_data_root_dir()
    out_data_dir = utils.get_output_dir()
# As an exemple, extract visual features for season 1, episode 1 of Friends
    #episode_path = root_data_dir + "algonauts_2025.competitors/stimuli/movies/friends/s1/friends_s01e01a.mkv"
    # Collecting the paths to all the movie stimuli
    file_in_filter = ''
    exclude_list = []#['friends_s03e05b', 'friends_s03e06a']
    files = glob(f"{root_data_dir}/algonauts_2025.competitors/stimuli/movies/friends/s3/*.mkv")

    if file_in_filter:
        files = [f for f in files if file_in_filter in f]
    files.sort()

    stimuli = {f.split("/")[-1].split(".")[0]: f for f in files}
    print(len(stimuli), list(stimuli)[:3], list(stimuli)[-3:])

    # Duration of each movie chunk, aligned with the fMRI TR of 1.49 seconds
    tr = 1.49

    

    # Saving directories
    save_dir_temp = utils.get_tmp_dir()
    hf_path = utils.get_mvl_model()
    device_map = split_model(hf_path)
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
        trust_remote_code=True,
        device_map=device_map).eval()
    tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True, use_fast=False)
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

    # Set up hooks once before processing all files
    hooks, layer_outputs, hook_storage = create_layer_hooks(model, custom_layers)
    
    try:
        # iterate across all the stimuli movie files
        iterator = tqdm(enumerate(stimuli.items()), total=len(list(stimuli)))
        for i, (stim_id, stim_path) in iterator:
            print(f"Extracting visual features for {stim_id}", stim_path)
            if stim_id in exclude_list:
                continue
            transcript_file = stim_path.replace('.mkv', '.tsv').replace('movies', 'transcripts')
            # Pass layer_outputs to the extraction function
            extract_visual_features(stim_id, stim_path, transcript_file, model, tokenizer, 
                                 layer_outputs, tr, save_dir_temp)
    finally:
        # Clean up hooks after all processing is done
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

def extract_video_chucks():
    root_data_dir = utils.get_data_root_dir()
    out_data_dir = os.path.join(utils.get_output_dir(), 'video_chunks')
    os.makedirs(out_data_dir, exist_ok=True)
    tr = 1.49
    seasons = ['s1','s2','s3','s4','s5','s6']
    for season in seasons:
        file_in_filter = ''
        exclude_list = []#['friends_s03e05b', 'friends_s03e06a']
        files = glob(f"{root_data_dir}/algonauts_2025.competitors/stimuli/movies/friends/{season}/*.mkv")

        if file_in_filter:
            files = [f for f in files if file_in_filter in f]
        files.sort()

        stimuli = {f.split("/")[-1].split(".")[0]: f for f in files}
        print(len(stimuli), list(stimuli)[:3], list(stimuli)[-3:])
        season_out_dir = os.path.join(out_data_dir, season)
        os.makedirs(season_out_dir, exist_ok=True)
        for stim_id, stim_path in stimuli.items():
            if stim_id in exclude_list:
                continue
            extract_save_video_chunks(stim_path, season_out_dir, stim_id, tr)


def extract_save_video_chunks(episode_path, save_dir, stim_id, tr):
    clip = VideoFileClip(episode_path)
    start_times = [x for x in np.arange(0, clip.duration, tr)][:-1]
    counter = 0
    px_save_dir = os.path.join(save_dir, 'px')
    with tqdm(total=len(start_times), desc="Extracting visual features") as pbar:
        for start in start_times:
            clip_chunk = clip.subclip(start, start+tr)
            chunk_path = os.path.join(save_dir, f'{stim_id}_tr_{counter}.mp4')
            clip_chunk.write_videofile(chunk_path, verbose=False, audio=False,
                logger=None)
            #pixel_values, num_patches_list = load_video(chunk_path, num_segments=8, max_num=1)
            #save_pixel_values(pixel_values, save_dir, f'{stim_id}_tr_{counter}')
            counter += 1
            pbar.update(1)
            # Divide the movie in chunks of length TR, and save the resulting	

def extract_visual_features(episode_id, episode_path, transcript_file, model, tokenizer, 
                          layer_outputs, tr, save_dir_temp):
    
    # Get the onset time of each movie chunk
    clip = VideoFileClip(episode_path)
    start_times = [x for x in np.arange(0, clip.duration, tr)][:-1]
    # Create the directory where the movie chunks are temporarily saved
    temp_dir = save_dir_temp # os.path.join(save_dir_temp, 'temp')
    #os.makedirs(temp_dir, exist_ok=True)
    # Empty features list
    extracted_features = []
    counter = 0
    n_used_words = 1000
    df = pd.read_csv(transcript_file, sep='\t').fillna("")
    trans_dataset = SentenceDataset(df["text_per_tr"].tolist(), mode="n_used_words", n_used_words=n_used_words)
    len_trans_dataset = len(trans_dataset)
    if len(trans_dataset) != len(start_times):
        print('clip.duration', clip.duration, start_times[0], start_times[-1])
    #assert len(trans_dataset) == len(start_times), f"len(dataset) = {len(trans_dataset)} != len(start_times) = {len(start_times)}"	
    # Loop over chunks
    with tqdm(total=len(start_times), desc="Extracting visual features") as pbar:
        for start in start_times:
            embeddings_dir = os.path.join(utils.get_output_dir(), 'embeddings')
            embeddings_prefix = f"{episode_id}_tr_{counter}"
            meta_file = os.path.join(embeddings_dir, f"{embeddings_prefix}_metadata.json")
            if not os.path.exists(meta_file):
                # Divide the movie in chunks of length TR, and save the resulting
                # clips as '.mp4' files
                clip_chunk = clip.subclip(start, start+tr)
                chunk_path = os.path.join(temp_dir, 'visual_chunk.mp4')
                clip_chunk.write_videofile(chunk_path, verbose=False, audio=False,
                    logger=None)
                # Load the frames from the chunked movie clip
                pixel_values, num_patches_list = load_video(chunk_path, num_segments=8, max_num=1)
                pixel_values = pixel_values.to(torch.bfloat16).cuda()
                video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
                trans_index = counter
                if trans_index >= len_trans_dataset:
                    trans_index = len_trans_dataset - 1
                question_for_embeddings = video_prefix + trans_dataset[trans_index]
                # Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
                #print('question_for_embeddings:', question_for_embeddings)
                #if meta file exists, skip the extraction
                # if not utils.isMockMode() or counter == 0:
                    # Use the existing hooks to get embeddings
                extracted_features = get_embeddings_with_existing_hooks(
                    model, 
                    tokenizer, 
                    pixel_values, 
                    question_for_embeddings,
                    layer_outputs
                )

                
                save_embeddings(extracted_features, embeddings_dir, text=trans_dataset[counter], 
                          prefix=embeddings_prefix)
            counter += 1
            pbar.update(1)


#process_all_files_for_extraction()
episode_path = "/home/bagga005/algo/comp_data/algonauts_2025.competitors/stimuli/movies/friends/s3/friends_s03e06a.mkv"
save_dir = "/home/bagga005/algo/comp_data/tmp/vid"
stim_id = "friends_s03e06a"
tr = 1.49
extract_video_chucks()
#extract_save_video_chunks(episode_path, save_dir, stim_id, tr)
exit()
