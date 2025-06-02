from decord import VideoReader, cpu
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import numpy as np
import requests
from moviepy.editor import VideoFileClip
from PIL import Image
import os
from glob import glob
import utils
from tqdm import tqdm

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

def extract_video_chucks():
    root_data_dir = utils.get_data_root_dir()
    out_data_dir = os.path.join(utils.get_output_dir(), 'video_chunks')
    os.makedirs(out_data_dir, exist_ok=True)
    tr = 1.49
    seasons = ['s4']
    for season in seasons:
        file_in_filter = ''
        exclude_list = []#['friends_s03e05b', 'friends_s03e06a']
        files = glob(f"{root_data_dir}/algonauts_2025.competitors/stimuli/movies/friends/{season}/*.mkv")

        if file_in_filter:
            files = [f for f in files if file_in_filter in f]
        files.sort(reverse=True)

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
    with tqdm(total=len(start_times), desc="Extracting video chunks for {}".format(stim_id)) as pbar:
        for start in start_times:
            chunk_path = os.path.join(save_dir, f'{stim_id}_tr_{counter}.mp4')
            if not os.path.exists(chunk_path):
                clip_chunk = clip.subclip(start, start+tr)
                clip_chunk.write_videofile(chunk_path, verbose=False, audio=False,
                    logger=None)
            #pixel_values, num_patches_list = load_video(chunk_path, num_segments=8, max_num=1)
            #save_pixel_values(pixel_values, save_dir, f'{stim_id}_tr_{counter}')
            counter += 1
            pbar.update(1)
            # Divide the movie in chunks of length TR, and save the resulting	

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
