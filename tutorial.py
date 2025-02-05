import os
from pathlib import Path
import glob
import re
import numpy as np
import pandas as pd
import h5py
import torch
import librosa
import ast
import string
import zipfile
from tqdm.notebook import tqdm
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import cv2
import nibabel as nib
from nilearn import plotting
from nilearn.maskers import NiftiLabelsMasker

import ipywidgets as widgets
from ipywidgets import VBox, Dropdown, Button
from IPython.display import Video, display, clear_output
from moviepy.editor import VideoFileClip
from transformers import BertTokenizer, BertModel
from torchvision.transforms import Compose, Lambda, CenterCrop
from torchvision.models.feature_extraction import create_feature_extractor
import utils



def load_mkv_file(movie_path):
    """
    Load video and audio data from the given .mkv movie file, and additionally
    prints related information.

    Parameters
    ----------
    movie_path : str
        Path to the .mkv movie file.

    """

    # Read the .mkv file
    cap = cv2.VideoCapture(movie_path)

    if not cap.isOpened():
        print("Error: Could not open movie.")
        return

    # Get video information
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = video_total_frames / video_fps
    video_duration_minutes = video_duration / 60

    # Print video information
    print(">>> Video Information <<<")
    print(f"Video FPS: {video_fps}")
    print(f"Video Resolution: {video_width}x{video_height}")
    print(f"Total Frames: {video_total_frames}")
    print(f"Video Duration: {video_duration:.2f} seconds or {video_duration_minutes:.2f} minutes")

    # Release the video object
    cap.release()

    # Audio information
    clip = VideoFileClip(movie_path)
    audio = clip.audio
    audio_duration = audio.duration
    audio_fps = audio.fps
    print("\n>>> Audio Information <<<")
    print(f"Audio Duration: {audio_duration:.2f} seconds")
    print(f"Audio FPS (Sample Rate): {audio_fps} Hz")

    # Extract and display the first 20 seconds of the video
    output_video_path = 'first_20_seconds.mp4'
    video_segment = clip.subclip(0, min(20, video_duration))
    print("\nCreating clip of the first 20 seconds of the video...")
    video_segment.write_videofile(output_video_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)

    # Display the video in the notebook
    if os.environ.get('WSL_DISTRO_NAME'):
        # For WSL environment, use xdg-open to open in default Windows video player
        print(f"\nVideo saved to: {output_video_path}")
        windows_path = output_video_path.replace('/', r'\\')
        os.system(f"explorer.exe {windows_path}")
    else:
        # Regular notebook display
        display(Video(output_video_path, embed=True, width=640, height=480))

def define_frames_transform():
    """Defines the preprocessing pipeline for the video frames. Note that this
    transform is specific to the slow_r50 model."""
    
    # Custom temporal subsample function to replace UniformTemporalSubsample
    def temporal_subsample(x, num_samples):
        t = x.shape[1]
        indices = torch.linspace(0, t-1, num_samples).long()
        return x[:, indices, :, :]
    
    # Custom short side scale function to replace ShortSideScale
    def short_side_scale(x, size):
        h, w = x.shape[-2:]
        if h < w:
            new_h = size
            new_w = int(size * w / h)
        else:
            new_w = size
            new_h = int(size * h / w)
        return torch.nn.functional.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
    
    # Custom normalize function
    def normalize(x):
        mean = torch.tensor([0.45, 0.45, 0.45]).view(-1, 1, 1, 1)
        std = torch.tensor([0.225, 0.225, 0.225]).view(-1, 1, 1, 1)
        return (x - mean) / std
    
    transform = Compose([
        Lambda(lambda x: temporal_subsample(x, 8)),  # Replace UniformTemporalSubsample
        Lambda(lambda x: x/255.0),
        Lambda(normalize),  # Replace Normalize
        Lambda(lambda x: short_side_scale(x, 256)),  # Replace ShortSideScale
        CenterCrop(256)
    ])
    return transform

def get_vision_model(device):
    """
    Load a pre-trained slow_r50 video model and set up the feature extractor.

    Parameters
    ----------
    device : torch.device
        The device on which the model will run (i.e., 'cpu' or 'cuda').

    Returns
    -------
    feature_extractor : torch.nn.Module
        The feature extractor model.
    model_layer : str
        The layer from which visual features will be extracted.

    """

    # Load the model
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50',
        pretrained=True)

    # Select 'blocks.5.pool' as the feature extractor layer
    model_layer = 'blocks.5.pool'
    feature_extractor = create_feature_extractor(model,
        return_nodes=[model_layer])
    feature_extractor.to(device)
    feature_extractor.eval()

    return feature_extractor, model_layer



def extract_visual_features(episode_path, tr, feature_extractor, model_layer,
    transform, device, save_dir_temp, save_dir_features):
    """
    Extract visual features from a movie using a pre-trained video model.

    Parameters
    ----------
    episode_path : str
        Path to the movie file for which the visual features are extracted.
    tr : float
        Duration of each chunk, in seconds (aligned with the fMRI repetition
        time, or TR).
    feature_extractor : torch.nn.Module
        Pre-trained feature extractor model.
    model_layer : str
        The model layer from which the visual features are extracted.
    transform : torchvision.transforms.Compose
        Transformation pipeline for processing video frames.
    device : torch.device
        Device for computation ('cpu' or 'cuda').
    save_dir_temp : str
        Directory where the chunked movie clips are temporarily stored for
        feature extraction.
    save_dir_features : str
        Directory where the extracted visual features are saved.

    Returns
    -------
    visual_features : float
        Array containing the extracted visual features.

    """

    # Get the onset time of each movie chunk
    clip = VideoFileClip(episode_path)
    start_times = [x for x in np.arange(0, clip.duration, tr)][:-1]
    # Create the directory where the movie chunks are temporarily saved
    temp_dir = os.path.join(save_dir_temp, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    # Empty features list
    visual_features = []
    # Loop over chunks
    with tqdm(total=len(start_times), desc="Extracting visual features") as pbar:
        for start in start_times:

            # Divide the movie in chunks of length TR, and save the resulting
            # clips as '.mp4' files
            clip_chunk = clip.subclip(start, start+tr)
            chunk_path = os.path.join(temp_dir, 'visual_chunk.mp4')
            clip_chunk.write_videofile(chunk_path, verbose=False, audio=False,
                logger=None)
            # Load the frames from the chunked movie clip
            video_clip = VideoFileClip(chunk_path)
            chunk_frames = [frame for frame in video_clip.iter_frames()]

            # Format the frames to shape:
            # (batch_size, channels, num_frames, height, width)
            frames_array = np.transpose(np.array(chunk_frames), (3, 0, 1, 2))
            print('frames_array.shape', frames_array.shape)
            # Convert the video frames to tensor
            inputs = torch.from_numpy(frames_array).float()
            print('inputs.shape', inputs.shape)
            # Preprocess the video frames
            inputs = transform(inputs).unsqueeze(0).to(device)
            print('inputs.shape', inputs.shape)
            # Extract the visual features
            with torch.no_grad():
                preds = feature_extractor(inputs)
            print('preds.shape', preds.shape)
            visual_features.append(np.reshape(preds[model_layer].cpu().numpy(), -1))
            # Update the progress bar
            pbar.update(1)

    # Convert the visual features to float32
    visual_features = np.array(visual_features, dtype='float32')

    # Save the visual features
    #out_file_visual = os.path.join(
    #    save_dir_features, f'friends_s01e01a_features_visual.h5')
    #with h5py.File(out_file_visual, 'a' if Path(out_file_visual).exists() else 'w') as f:
    #    group = f.create_group("s01e01a")
    #    group.create_dataset('visual', data=visual_features, dtype=np.float32)
    #print(f"Visual features saved to {out_file_visual}")

    # Output
    return visual_features

def main():
    transform = define_frames_transform()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    root_data_dir = utils.get_data_root_dir()
    print(root_data_dir)
    # movie_path = root_data_dir + "/algonauts_2025.competitors/stimuli/movies/friends/s1/friends_s01e01a.mkv"
    # print(movie_path)
    # load_mkv_file(movie_path)
    # As an exemple, extract visual features for season 1, episode 1 of Friends
    feature_extractor, model_layer = get_vision_model(device)
    episode_path = root_data_dir + "/algonauts_2025.competitors/stimuli/movies/friends/s1/friends_s01e01a.mkv"
    print(episode_path)

    # Duration of each movie chunk, aligned with the fMRI TR of 1.49 seconds
    tr = 1.49

    # Saving directories
    save_dir_temp = "./visual_features"
    save_dir_features = root_data_dir +  "/stimulus_features/raw/visual/"

    # Execute visual feature extraction
    visual_features = extract_visual_features(episode_path, tr, feature_extractor,
        model_layer, transform, device, save_dir_temp, save_dir_features)
    
    print("Visual features shape for 'friends_s01e01a.mkv':")
    print(visual_features.shape)
    print('(Movie samples × Visual features length)')

    # Visualize the features for five movie chunks
    print("\nVisual feature vectors for 5 movie chunks:\n")
    print(visual_features[20:25])

if __name__ == "__main__":
    main()