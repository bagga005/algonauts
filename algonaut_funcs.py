import os
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from torchvision.transforms import Compose, Lambda, CenterCrop
from torchvision.models.feature_extraction import create_feature_extractor
from pytorchvideo.transforms import Normalize, UniformTemporalSubsample, ShortSideScale
import utils
from transformers import BertTokenizer, BertModel
import librosa
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import h5py
import pandas as pd
import string
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def uniform_temporal_subsample(x: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Uniformly samples num_samples frames from a video.
    
    Args:
        x: Video tensor of shape (C, T, H, W)
        num_samples: The number of samples to take
    
    Returns:
        Tensor of shape (C, num_samples, H, W)
    """
    t = x.shape[1]
    indices = torch.linspace(0, t - 1, num_samples)
    indices = torch.clamp(indices, 0, t - 1).long()
    return torch.index_select(x, 1, indices)

def define_frames_transform():
    """Defines the preprocessing pipeline for the video frames. Note that this
    transform is specific to the slow_r50 model."""
    transform = Compose(
        [
            UniformTemporalSubsample(8),
            #Lambda(lambda x: uniform_temporal_subsample(x, num_samples=8)),
            Lambda(lambda x: x/255.0),
            Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
            ShortSideScale(size=256),
            # Lambda(lambda x: Resize(size=256, antialias=True)(x) if x.shape[-2] < x.shape[-1] else 
            #       Resize(size=(int(256 * x.shape[-2]/x.shape[-1]), 256), antialias=True)(x)),
            CenterCrop(256)
        ]
  )
    return transform

transform = define_frames_transform()

def get_vision_model():
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the model
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50',
        pretrained=True)

    # Select 'blocks.5.pool' as the feature extractor layer
    model_layer = 'blocks.5.pool'
    feature_extractor = create_feature_extractor(model,
        return_nodes=[model_layer])
    feature_extractor.to(device)
    feature_extractor.eval()

    return feature_extractor, model_layer, device

def extract_visual_preprocessed_features(episode_path, tr,
    transform, save_dir_temp, save_file, group_name):


    # Get the onset time of each movie chunk
    clip = VideoFileClip(episode_path)
    start_times = [x for x in np.arange(0, clip.duration, tr)][:-1]
    # Create the directory where the movie chunks are temporarily saved
    temp_dir = save_dir_temp # os.path.join(save_dir_temp, 'temp')
    #os.makedirs(temp_dir, exist_ok=True)
    # Empty features list
    visual_features = []
    tr_index = 0

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
            # Convert the video frames to tensor
            inputs = torch.from_numpy(frames_array).float()
            #print('input 1', inputs.shape)
            # Preprocess the video frames
            inputs = transform(inputs).unsqueeze(0).cpu()
            #print('input 2', inputs.shape)
            #inputs = np.reshape(inputs.numpy(), -1)
            #print('input 3', inputs.shape)
            visual_features.append(inputs)
            #print('visual_features', visual_features.shape)

            # Update the progress bar
            pbar.update(1)
            tr_index += 1
    # Convert the visual features to float32
    
    visual_features = np.array(visual_features, dtype='float32')
    # Save the visual features
    with h5py.File(save_file, 'a' if Path(save_file).exists() else 'w') as f:
        group = f.create_group(group_name)
        group.create_dataset('visual', data=visual_features, dtype=np.float32)
    #print('visual_features', visual_features.shape)
    # Output
    return visual_features

def extract_visual_features_from_preprocessed_video(episode_path, feature_extractor, model_layer, device, save_file, group_name):
    #load the preprocessed video
    with h5py.File(episode_path, 'r') as f1:
        print("Root level keys:", list(f1.keys()))
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
            # Convert the video frames to tensor
            inputs = torch.from_numpy(frames_array).float()
            # Preprocess the video frames
            inputs = transform(inputs).unsqueeze(0).to(device)

            # Extract the visual features
            with torch.no_grad():
                preds = feature_extractor(inputs)
            visual_features.append(np.reshape(preds[model_layer].cpu().numpy(), -1))

            # Update the progress bar
            pbar.update(1)

    # Convert the visual features to float32
    
    visual_features = np.array(visual_features, dtype='float32')
    # Save the visual features
    with h5py.File(save_file, 'a' if Path(save_file).exists() else 'w') as f:
        group = f.create_group(group_name)
        group.create_dataset('visual', data=visual_features, dtype=np.float32)
    print('visual_features', visual_features.shape)
    # Output
    return visual_features



def extract_visual_features(episode_path, tr, feature_extractor, model_layer,
    transform, device, save_dir_temp, save_file, group_name):
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
    temp_dir = save_dir_temp # os.path.join(save_dir_temp, 'temp')
    #os.makedirs(temp_dir, exist_ok=True)
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
            # Convert the video frames to tensor
            inputs = torch.from_numpy(frames_array).float()
            # Preprocess the video frames
            inputs = transform(inputs).unsqueeze(0).to(device)

            # Extract the visual features
            with torch.no_grad():
                preds = feature_extractor(inputs)
            visual_features.append(np.reshape(preds[model_layer].cpu().numpy(), -1))

            # Update the progress bar
            pbar.update(1)

    # Convert the visual features to float32
    
    visual_features = np.array(visual_features, dtype='float32')
    # Save the visual features
    with h5py.File(save_file, 'a' if Path(save_file).exists() else 'w') as f:
        group = f.create_group(group_name)
        group.create_dataset('visual', data=visual_features, dtype=np.float32)
    print('visual_features', visual_features.shape)
    # Output
    return visual_features

def load_features_visual(path, stim_id):
    """
    Load the extracted features from the HDF5 file.

    Parameters
    ----------
    root_data_dir : str
        Root data directory.
    modality : str
        The modality of the features ('visual', 'audio', or 'language').

    Returns
    -------
    features : float
        Stimulus features.

    """

    ### Get the stimulus features file directory ###
    data_dir = os.path.join(path, stim_id+'.h5')
        #                     'stimulus_features', 'raw', modality,
        # 'friends_s01e01a_features_'+modality+'.h5')

    ### Load the stimulus features ###
    with h5py.File(data_dir, 'r') as data:
        assert len(data.keys()) == 1, f"Expected exactly 1 key, but found {len(data.keys())} keys"
        features = np.asarray(data[stim_id]['visual'])

    print(f"visual features original shape: {features.shape}")
    print('(Movie samples × Features)')

    ### Output ###
    return features

def extract_audio_features(episode_path, tr, sr, save_dir_temp,
    save_file, group_name):
    """
    Extract audio features from a movie using Mel-frequency cepstral
    coefficients (MFCCs).

    Parameters
    ----------
    episode_path : str
        Path to the movie file for which the audio features are extracted.
    tr : float
        Duration of each chunk, in seconds (aligned with the fMRI repetition
        time, or TR).
    sr : int
        Audio sampling rate.
    device : str
        Device to perform computations ('cpu' or 'gpu').
    save_dir_temp : str
        Directory where the chunked movie clips are temporarily stored for
        feature extraction.
    save_dir_features : str
        Directory where the extracted audio features are saved.

    Returns
    -------
    audio_features : float
        Array containing the extracted audio features.

    """

    # Get the onset time of each movie chunk
    clip = VideoFileClip(episode_path)
    start_times = [x for x in np.arange(0, clip.duration, tr)][:-1]
    dir_ext = random.randint(1, 10000000)
    # Create the directory where the movie chunks are temporarily saved
    temp_dir = os.path.join(save_dir_temp, 'temp' + str(dir_ext))
    os.makedirs(temp_dir, exist_ok=True)

    # Empty features list
    audio_features = []

    ### Loop over chunks ###
    with tqdm(total=len(start_times), desc="Extracting audio features") as pbar:
        for start in start_times:

            # Divide the movie in chunks of length TR, and save the resulting
            # audio clips as '.wav' files
            clip_chunk = clip.subclip(start, start+tr)
            chunk_audio_path = os.path.join(temp_dir, 'audio_s01e01a.wav')
            clip_chunk.audio.write_audiofile(chunk_audio_path, verbose=False,
                logger=None)
            # Load the audio samples from the chunked movie clip
            y, sr = librosa.load(chunk_audio_path, sr=sr, mono=True)

            # Extract the audio features (MFCC)
            mfcc_features = np.mean(librosa.feature.mfcc(y=y, sr=sr), axis=1)
            audio_features.append(mfcc_features)
            # Update the progress bar
            pbar.update(1)

    ### Convert the visual features to float32 ###
    audio_features = np.array(audio_features, dtype='float32')

    # Save the visual features
    with h5py.File(save_file, 'a' if Path(save_file).exists() else 'w') as f:
        group = f.create_group(group_name)
        group.create_dataset('audio', data=audio_features, dtype=np.float32)

    # Save the audio features
    #out_file_audio = os.path.join(
    #    save_dir_features, f'friends_s01e01a_features_audio.h5')
    #with h5py.File(out_file_audio, 'a' if Path(out_file_audio).exists() else 'w') as f:
    #    group = f.create_group("s01e01a")
    #    group.create_dataset('audio', data=audio_features, dtype=np.float32)
    #print(f"Audio features saved to {out_file_audio}")

    ### Output ###
    return audio_features

def get_language_model(device):
    """
    Load a pre-trained bert-base-uncased language model and its corresponding
    tokenizer.

    Parameters
    ----------
    device : torch.device
        Device on which the model will run (e.g., 'cpu' or 'cuda').

    Returns
    -------
    model : object
        Pre-trained language model.
    tokenizer : object
        Tokenizer corresponding to the language model.

    """

    ### Load the model ###
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval().to(device)

    ### Load the tokenizer ###
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
        do_lower_case=True)

    ### Output ###
    return model, tokenizer

# Load the model and tokenizer
model, tokenizer = get_language_model(device)

def extract_language_features(episode_path, model, tokenizer, num_used_tokens,
    kept_tokens_last_hidden_state, device,  save_file, group_name):
    """
    Extract language features from a movie using a pre-trained language model.

    Parameters
    ----------
    episode_path : str
        Path to the movie transcripts for which the language features are
        extracted.
    model : object
        Pre-trained language model.
    tokenizer : object
        Tokenizer corresponding to the language model.
    num_used_tokens : int
        Total number of tokens that are fed to the language model for each
        chunk, including the tokens from the chunk of interest plus N tokens
        from previous chunks (the maximum allowed by the model is 510).
    kept_tokens_last_hidden_state : int
        Number of features retained for the last_hidden_state, where each
        feature corresponds to a token, starting from the most recent token.
    device : str
        Device to perform computations ('cpu' or 'gpu').
    save_dir_features : str
        Directory where the extracted language features are saved.

    Returns
    -------
    pooler_output : list
        List containing the pooler_output features for each chunk.
    last_hidden_state : list
        List containing the last_hidden_state features for each chunk

    """

    ### Load the transcript ###
    df = pd.read_csv(episode_path, sep='\t')
    df.insert(loc=0, column="is_na", value=df["text_per_tr"].isna())

    ### Initialize the tokens and features lists ###
    tokens, np_tokens, pooler_output, last_hidden_state = [], [], [], []

    ### Loop over text chunks ###
    for i in tqdm(range(df.shape[0]), desc="Extracting language features"):

        ### Tokenize raw text ###
        if not df.iloc[i]["is_na"]: # Only tokenize if words were spoken during a chunk (i.e., if the chunk is not empty)
            # Tokenize raw text with puntuation (for pooler_output features)
            tr_text = df.iloc[i]["text_per_tr"]
            tokens.extend(tokenizer.tokenize(tr_text))
            # Tokenize without punctuation (for last_hidden_state features)
            tr_np_tokens = tokenizer.tokenize(
                tr_text.translate(str.maketrans('', '', string.punctuation)))
            np_tokens.extend(tr_np_tokens)

        ### Extract the pooler_output features ###
        if len(tokens) > 0: # Only extract features if there are tokens available
            # Select the number of tokens used from the current and past chunks,
            # and convert them into IDs
            used_tokens = tokenizer.convert_tokens_to_ids(
                tokens[-(num_used_tokens):])
            # IDs 101 and 102 are special tokens that indicate the beginning and
            # end of an input sequence, respectively.
            input_ids = [101] + used_tokens + [102]
            tensor_tokens = torch.tensor(input_ids).unsqueeze(0).to(device)
            # Extract and store the pooler_output features
            with torch.no_grad():
                outputs = model(tensor_tokens)
                pooler_output.append(outputs['pooler_output'][0].cpu().numpy())
        else: # Store NaN values if no tokes are available
            pooler_output.append(np.full(768, np.nan, dtype='float32'))

        ### Extract the last_hidden_state features ###
        if len(np_tokens) > 0: # Only extract features if there are tokens available
            np_feat = np.full((kept_tokens_last_hidden_state, 768), np.nan, dtype='float32')
            # Select the number of tokens used from the current and past chunks,
            # and convert them into IDs
            used_tokens = tokenizer.convert_tokens_to_ids(
                np_tokens[-(num_used_tokens):])
            # IDs 101 and 102 are special tokens that indicate the beginning and
            # end of an input sequence, respectively.
            np_input_ids = [101] + used_tokens + [102]
            np_tensor_tokens = torch.tensor(np_input_ids).unsqueeze(0).to(device)
            # Extract and store the last_hidden_state features
            with torch.no_grad():
                np_outputs = model(np_tensor_tokens)
                np_outputs = np_outputs['last_hidden_state'][0][1:-1].cpu().numpy()
            tk_idx = min(kept_tokens_last_hidden_state, len(np_tokens))
            np_feat[-tk_idx:, :] = np_outputs[-tk_idx:]
            last_hidden_state.append(np_feat)
        else: # Store NaN values if no tokens are available
            last_hidden_state.append(np.full(
                (kept_tokens_last_hidden_state, 768), np.nan, dtype='float32'))

    ### Convert the language features to float32 ###
    pooler_output = np.array(pooler_output, dtype='float32')
    last_hidden_state = np.array(last_hidden_state, dtype='float32')

    with h5py.File(save_file, 'a' if Path(save_file).exists() else 'w') as f:
        group = f.create_group(group_name)
        group.create_dataset('language_pooler_output', data=pooler_output,
           dtype=np.float32)
        group.create_dataset('language_last_hidden_state',
           data=last_hidden_state, dtype=np.float32)

    ### Save the language features ###
    #out_file_language = os.path.join(
    #    save_dir_features, f'friends_s01e01a_features_language.h5')
    #with h5py.File(out_file_language, 'a' if Path(out_file_language).exists() else 'w') as f:
    #    group = f.create_group("s01e01a")
    #    group.create_dataset('language_pooler_output', data=pooler_output,
    #        dtype=np.float32)
    #    group.create_dataset('language_last_hidden_state',
    #        data=last_hidden_state, dtype=np.float32)
    #print(f"Language features saved to {out_file_language}")

    

    ### Output ###
    return pooler_output, last_hidden_state

def load_features(file_path, modality):
    """
    Load the extracted features from the HDF5 file.

    Parameters
    ----------
    root_data_dir : str
        Root data directory.
    modality : str
        The modality of the features ('visual', 'audio', or 'language').

    Returns
    -------
    features : float
        Stimulus features.

    """

    ### Get the stimulus features file directory ###
    # data_dir = os.path.join(root_data_dir, 'stimulus_features', 'raw', modality,
    #     'friends_s01e01a_features_'+modality+'.h5')

    ### Load the stimulus features ###
    with h5py.File(file_path, 'r') as data:
        # Method 1: Using assert statement
        # print(file_path)
        # print(data.keys())
        assert len(data.keys()) == 1, f"Expected exactly 1 key, but found {len(data.keys())} keys"

        for episode in data.keys():
            if modality != 'language':
                features = np.asarray(data[episode][modality])
                print(f"{episode} features original shape: {features.shape}")
            else:
                # Vectorize and append pooler_output and last_hidden_state
                # language features
                pooler_output = np.asarray(
                    data[episode][modality+'_pooler_output'])
                last_hidden = np.asarray(np.reshape(
                    data[episode][modality+'_last_hidden_state'],
                    (len(pooler_output), -1)))
                features = np.append(pooler_output, last_hidden, axis=1)

    ### Output ###
    return features

def preprocess_features(features, zscore=True):
    """
    Rplaces NaN values in the stimulus features with zeros, and z-score the
    features.

    Parameters
    ----------
    features : float
        Stimulus features.

    Returns
    -------
    prepr_features : float
        Preprocessed stimulus features.

    """

   ### Convert NaN values to zeros ###
    features = np.nan_to_num(features)

    if zscore:
        ### Z-score the features ###
        scaler = StandardScaler()
        prepr_features = scaler.fit_transform(features)
    else:
        prepr_features = features

    ### Output ###
    return prepr_features
    
def perform_pca(prepr_features, n_components, modality):
    """
    Perform PCA on the standardized features.

    Parameters
    ----------
    prepr_features : float
        Preprocessed stimulus features.
    n_components : int
        Number of components to keep

    Returns
    -------
    features_pca : float
        PCA-downsampled stimulus features.

    """

    ### Set the number of principal components to keep ###
    # If number of PCs is larger than the number of features, set the PC number
    # to the number of features
    if n_components > prepr_features.shape[1]:
        n_components = prepr_features.shape[1]

    ### Perform PCA ###n_init=4, max_iter=300
    pca = PCA(n_components, random_state=1001, svd_solver="full")
    features_pca = pca.fit_transform(prepr_features)
    print("Var expl:", pca.explained_variance_ratio_.sum() )
    print(f"\n{modality} features PCA shape: {features_pca.shape}")
    print('(Movie samples × Principal components)')

    ### Output ###
    return features_pca