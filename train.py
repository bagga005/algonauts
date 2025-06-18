import numpy as np
import math
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import h5py
import os
import utils
from tqdm import tqdm
from model_sklearn import LinearHandler_Sklearn
from model_torchregression import RegressionHander_Pytorch
from model_transformer import RegressionHander_Transformer
from model_lora_vision import RegressionHander_Vision
#from model_lora_vision import RegressionHander_Vision
import nibabel as nib
from nilearn import plotting
from nilearn.maskers import NiftiLabelsMasker
import time
import pandas as pd
import matplotlib.pyplot as plt
import json
import datetime
from roi_network_map import get_breakup_by_network
from algonaut_funcs import prepare_s7_fmri_for_alignment

def get_boundary_from_fmri_for_movie_for_subject(subject, movie_name, fmri=None):
    assert subject == None or fmri == None, "subject and fmri cannot be provided together"
    if movie_name == "friends-s07":
        if subject == None:
            subject = 1
        _, boundary = prepare_s7_fmri_for_alignment(subject)
        return boundary
    else:
        if movie_name[:7] == 'friends':
            id = movie_name[8:]
        elif movie_name[:7] == 'movie10':
            id = movie_name[8:]
        if fmri is None:
            fmri = get_fmri(subject)
        movie_splits = [key for key in fmri if id in key[:len(id)]]
        boundary = []
        for split in movie_splits:
            boundary.append((split, fmri[split].shape[0]))
        return boundary
    
def load_stimulus_features(root_data_dir, modality):
    """
    Load the stimulus features.

    Parameters
    ----------
    root_data_dir : str
        Root data directory.
    modality : str
        Used feature modality.

    Returns
    -------
    features : dict
        Dictionary containing the stimulus features.

    """

    features = {}

    ### Load the visual features ###
    if modality == 'visual' or modality == 'all' or modality == 'visual+language' or modality == 'visual+audio':
        stimuli_dir = os.path.join(root_data_dir, 'pca',
            'friends_movie10', 'visual', 'features_train.npy')
        features['visual'] = np.load(stimuli_dir, allow_pickle=True).item()

    ### Load the audio features ###
    if modality == 'audio' or modality == 'all' or modality == 'audio+language' or modality == 'visual+audio':
        stimuli_dir = os.path.join(root_data_dir, 'pca',
            'friends_movie10', 'audio', 'features_train.npy')
        features['audio'] = np.load(stimuli_dir, allow_pickle=True).item()

    ### Load the language features ###
    if modality == 'language' or modality == 'all' or modality == 'audio+language' or modality == 'visual+language':
        stimuli_dir = os.path.join(root_data_dir, 'pca',
            'friends_movie10', 'language', 'features_train.npy')
        features['language'] = np.load(stimuli_dir, allow_pickle=True).item()

    ### Output ###
    return features

def load_fmri(root_data_dir, subject):
    """
    Load the fMRI responses for the selected subject.

    Parameters
    ----------
    root_data_dir : str
        Root data directory.
    subject : int
        Subject used to train and validate the encoding model.

    Returns
    -------
    fmri : dict
        Dictionary containing the  fMRI responses.

    """

    fmri = {}

    ### Load the fMRI responses for Friends ###
    # Data directory
    fmri_file = f'sub-0{subject}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5'
    fmri_dir = os.path.join(root_data_dir, 'algonauts_2025.competitors',
        'fmri', f'sub-0{subject}', 'func', fmri_file)
    # Load the the fMRI responses
    fmri_friends = h5py.File(fmri_dir, 'r')
    for key, val in fmri_friends.items():
        # print('key', key)
        # print('val', val[:].shape)
        fmri[str(key[13:])] = val[:].astype(np.float32)
    del fmri_friends

    ### Load the fMRI responses for Movie10 ###
    # Data directory
    fmri_file = f'sub-0{subject}_task-movie10_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5'
    fmri_dir = os.path.join(root_data_dir, 'algonauts_2025.competitors',
        'fmri', f'sub-0{subject}', 'func', fmri_file)
    # Load the the fMRI responses
    fmri_movie10 = h5py.File(fmri_dir, 'r')
    for key, val in fmri_movie10.items():
        fmri[key[13:]] = val[:].astype(np.float32)
    del fmri_movie10
    # Average the fMRI responses across the two repeats for 'figures'
    keys_all = fmri.keys()
    figures_splits = 12
    for s in range(figures_splits):
        movie = 'figures' + format(s+1, '02')
        keys_movie = [rep for rep in keys_all if movie in rep]
        fmri[movie] = ((fmri[keys_movie[0]] + fmri[keys_movie[1]]) / 2).astype(np.float32)
        #fmri[movie] = (fmri[keys_movie[1]]).astype(np.float32)
        del fmri[keys_movie[0]]
        del fmri[keys_movie[1]]
    # Average the fMRI responses across the two repeats for 'life'
    keys_all = fmri.keys()
    life_splits = 5
    for s in range(life_splits):
        movie = 'life' + format(s+1, '02')
        keys_movie = [rep for rep in keys_all if movie in rep]
        fmri[movie] = ((fmri[keys_movie[0]] + fmri[keys_movie[1]]) / 2).astype(np.float32)
        #fmri[movie] = (fmri[keys_movie[0]]).astype(np.float32)
        del fmri[keys_movie[0]]
        del fmri[keys_movie[1]]

    ### Output ###
    return fmri

def get_fmri_for_all_subjects():
    fmri1 = get_fmri(1)
    fmri2 = get_fmri(2)
    fmri3 = get_fmri(3)
    fmri5 = get_fmri(5)
    fmri1_aligned = {}
    fmri2_aligned = {}
    fmri3_aligned = {}
    fmri5_aligned = {}
    
    for key in fmri5.keys():
        if key != 's05e20a' and key != 's06e03a':
            fmri5_aligned[key] = fmri5[key]
            fmri3_aligned[key] = fmri3[key]
            fmri2_aligned[key] = fmri2[key]
            fmri1_aligned[key] = fmri1[key]
    
    fmri = {}
    fmri['fmri1'] = fmri1_aligned
    fmri['fmri2'] = fmri2_aligned
    fmri['fmri3'] = fmri3_aligned
    fmri['fmri5'] = fmri5_aligned

    return fmri


def normalize_to_radians(value, original_min=0, original_max=49, target_min=-math.pi/2, target_max=math.pi/2):
    """
    Normalize a value from the original range to the target range.
    
    Parameters:
    -----------
    value : float
        The value to normalize
    original_min : float
        The minimum value of the original range
    original_max : float
        The maximum value of the original range
    target_min : float
        The minimum value of the target range
    target_max : float
        The maximum value of the target range
        
    Returns:
    --------
    float
        The normalized value in the target range
    """
    # First, normalize to [0, 1]
    normalized = (value - original_min) / (original_max - original_min)
    
    # Then, scale to target range
    scaled = normalized * (target_max - target_min) + target_min
    
    return scaled

def load_stimulus_features_friends_s7(root_data_dir):
    """
    Load the stimulus features of all modalities (visual + audio + language) for
    Friends season 7.

    Parameters
    ----------
    root_data_dir : str
        Root data directory.

    Returns
    -------
    features_friends_s7 : dict
        Dictionary containing the stimulus features for Friends season 7.

    """

    features_friends_s7 = {}

    ### Load the visual features ###
    stimuli_dir = os.path.join(root_data_dir, 'stimulus_features', 'pca',
        'friends_movie10', 'visual', 'features_test.npy')
    features_friends_s7['visual'] = np.load(stimuli_dir,
        allow_pickle=True).item()

    ### Load the audio features ###
    stimuli_dir = os.path.join(root_data_dir, 'stimulus_features', 'pca',
        'friends_movie10', 'audio', 'features_test.npy')
    features_friends_s7['audio'] = np.load(stimuli_dir,
        allow_pickle=True).item()

    ### Load the language features ###
    stimuli_dir = os.path.join(root_data_dir, 'stimulus_features', 'pca',
        'friends_movie10', 'language', 'features_test.npy')
    features_friends_s7['language'] = np.load(stimuli_dir,
        allow_pickle=True).item()

    ### Output ###
    return features_friends_s7

def align_features_and_fmri_samples_friends_s7(subject, features_friends_s7,
    root_data_dir):
    """
    Align the stimulus feature with the fMRI response samples for Friends season
    7 episodes, later used to predict the fMRI responses for challenge
    submission.

    Parameters
    ----------
    features_friends_s7 : dict
        Dictionary containing the stimulus features for Friends season 7.
    root_data_dir : str
        Root data directory.

    Returns
    -------
    aligned_features_friends_s7 : dict
        Aligned stimulus features for each subject and Friends season 7 episode.

    """

    ### Empty results dictionary ###
    aligned_features_friends_s7 = {}

    ### HRF delay ###
    # fMRI detects the BOLD (Blood Oxygen Level Dependent) response, a signal
    # that reflects changes in blood oxygenation levels in response to activity
    # in the brain. Blood flow increases to a given brain region in response to
    # its activity. This vascular response, which follows the hemodynamic
    # response function (HRF), takes time. Typically, the HRF peaks around 5–6
    # seconds after a neural event: this delay reflects the time needed for
    # blood oxygenation changes to propagate and for the fMRI signal to capture
    # them. Therefore, this parameter introduces a delay between stimulus chunks
    # and fMRI samples for a better correspondence between input stimuli and the
    # brain response. For example, with a hrf_delay of 3, if the stimulus chunk
    # of interest is 17, the corresponding fMRI sample will be 20.
    hrf_delay = 3

    ### Stimulus window ###
    # stimulus_window indicates how many stimulus feature samples are used to
    # model each fMRI sample, starting from the stimulus sample corresponding to
    # the fMRI sample of interest, minus the hrf_delay, and going back in time.
    # For example, with a stimulus_window of 5, and a hrf_delay of 3, if the
    # fMRI sample of interest is 20, it will be modeled with stimulus samples
    # [13, 14, 15, 16, 17]. Note that this only applies to visual and audio
    # features, since the language features were already extracted using
    # transcript words spanning several movie samples (thus, each fMRI sample
    # will only be modeled using the corresponding language feature sample,
    # minus the hrf_delay). Also note that a larger stimulus window will
    # increase compute time, since it increases the amount of stimulus features
    # used to train and validate the fMRI encoding models. Here you will use a
    # value of 5, since this is how the challenge baseline encoding models were
    # trained.
    stimulus_window = 5

    ### Loop over subjects ###
    subjects = [1, 2, 3, 5]
    desc = "Aligning stimulus and fMRI features of the four subjects"
    sub = subject
    aligned_features_friends_s7 = {}

    ### Load the Friends season 7 fMRI samples ###
    samples_dir = os.path.join(root_data_dir, 'algonauts_2025.competitors',
        'fmri', f'sub-0{sub}', 'target_sample_number',
        f'sub-0{sub}_friends-s7_fmri_samples.npy')
    fmri_samples = np.load(samples_dir, allow_pickle=True).item()
    total_dim = 0
    ### Loop over Friends season 7 episodes ###
    for epi, samples in fmri_samples.items():
        features_epi = []
        ### Loop over fMRI samples ###
        for s in range(samples):
            # Empty variable containing the stimulus features of all
            # modalities for each sample
            f_all = np.empty(0)

            ### Loop across modalities ###
            for mod in features_friends_s7.keys():

                ### Visual and audio features ###
                # If visual or audio modality, model each fMRI sample using
                # the N stimulus feature samples up to the fMRI sample of
                # interest minus the hrf_delay (where N is defined by the
                # 'stimulus_window' variable)
                if mod == 'visual' or mod == 'audio':
                    # In case there are not N stimulus feature samples up to
                    # the fMRI sample of interest minus the hrf_delay (where
                    # N is defined by the 'stimulus_window' variable), model
                    # the fMRI sample using the first N stimulus feature
                    # samples
                    if s < (stimulus_window + hrf_delay):
                        idx_start = 0
                        idx_end = idx_start + stimulus_window
                    else:
                        idx_start = s - hrf_delay - stimulus_window + 1
                        idx_end = idx_start + stimulus_window
                    # In case there are less visual/audio feature samples
                    # than fMRI samples minus the hrf_delay, use the last N
                    # visual/audio feature samples available (where N is
                    # defined by the 'stimulus_window' variable)
                    if idx_end > len(features_friends_s7[mod][epi]):
                        idx_end = len(features_friends_s7[mod][epi])
                        idx_start = idx_end - stimulus_window
                    f = features_friends_s7[mod][epi][idx_start:idx_end]
                    f_all = np.append(f_all, f.flatten())
                   # total_dim += f.shape[0]
                ### Language features ###
                # Since language features already consist of embeddings
                # spanning several samples, only model each fMRI sample
                # using the corresponding stimulus feature sample minus the
                # hrf_delay
                elif mod == 'language':
                    # In case there are no language features for the fMRI
                    # sample of interest minus the hrf_delay, model the fMRI
                    # sample using the first language feature sample
                    if s < hrf_delay:
                        idx = 0
                    else:
                        idx = s - hrf_delay
                    # In case there are fewer language feature samples than
                    # fMRI samples minus the hrf_delay, use the last
                    # language feature sample available
                    if idx >= (len(features_friends_s7[mod][epi]) - hrf_delay):
                        f = features_friends_s7[mod][epi][-1,:]
                    else:
                        f = features_friends_s7[mod][epi][idx]
                    f_all = np.append(f_all, f.flatten())
                    #total_dim += f.shape[0]
            ### Append the stimulus features of all modalities for this sample ###
            features_epi.append(f_all)
        total_dim += len(features_epi)
        ### Add the episode stimulus features to the features dictionary ###
        aligned_features_friends_s7[epi] = np.asarray(
            features_epi, dtype=np.float32)
    print(f"Total dimension for subject {sub}: {total_dim}")

    return aligned_features_friends_s7

def do_features_fmri_len_check(features, fmri, movie_name):
    #do based on subject 1
     boundary = get_boundary_from_fmri_for_movie_for_subject(None, movie_name, fmri)
     assert len(boundary) > 4, f"boundary cant be so small len(boundary) {len(boundary)} for movie {movie_name}"

     for stim_id, size in boundary:
            passed = (size == len(features[stim_id]) or size == len(features[stim_id])+1) or \
                (size == len(features[stim_id])+2) or (stim_id in ['bourne01', 'life01' , 'life02', 'life03', 'life04', 'life05'])
            if not passed:
                print(f"Mismatch: for movie {movie_name} and stim_id {stim_id}. fmri size {size} != feature size {len(features[stim_id])} ")


def align_features_and_fmri_samples(features, fmri, excluded_samples_start,
    excluded_samples_end, hrf_delay, stimulus_window, movies, viewing_session, summary_features=False, all_subject_fmri=False):
    """
    Align the stimulus feature with the fMRI response samples for the selected
    movies, later used to train and validate the encoding models.

    Parameters
    ----------
    features : dict
        Dictionary containing the stimulus features.
    fmri : dict
        Dictionary containing the fMRI responses.
    excluded_trs_start : int
        Integer indicating the first N fMRI TRs that will be excluded and not
        used for model training. The reason for excluding these TRs is that due
        to the latency of the hemodynamic response the fMRI responses of first
        few fMRI TRs do not yet contain stimulus-related information.
    excluded_trs_end : int
        Integer indicating the last N fMRI TRs that will be excluded and not
        used for model training. The reason for excluding these TRs is that
        stimulus feature samples (i.e., the stimulus chunks) can be shorter than
        the fMRI samples (i.e., the fMRI TRs), since in some cases the fMRI
        run ran longer than the actual movie. However, keep in mind that the fMRI
        timeseries onset is ALWAYS SYNCHRONIZED with movie onset (i.e., the
        first fMRI TR is always synchronized with the first stimulus chunk).
    hrf_delay : int
        fMRI detects the BOLD (Blood Oxygen Level Dependent) response, a signal
        that reflects changes in blood oxygenation levels in response to
        activity in the brain. Blood flow increases to a given brain region in
        response to its activity. This vascular response, which follows the
        hemodynamic response function (HRF), takes time. Typically, the HRF
        peaks around 5–6 seconds after a neural event: this delay reflects the
        time needed for blood oxygenation changes to propagate and for the fMRI
        signal to capture them. Therefore, this parameter introduces a delay
        between stimulus chunks and fMRI samples for a better correspondence
        between input stimuli and the brain response. For example, with a
        hrf_delay of 3, if the stimulus chunk of interest is 17, the
        corresponding fMRI sample will be 20.
    stimulus_window : int
        Integer indicating how many stimulus features' chunks are used to model
        each fMRI TR, starting from the chunk corresponding to the TR of
        interest, and going back in time. For example, with a stimulus_window of
        5, if the fMRI TR of interest is 20, it will be modeled with stimulus
        chunks [16, 17, 18, 19, 20]. Note that this only applies to visual and
        audio features, since the language features were already extracted using
        transcript words spanning several movie chunks (thus, each fMRI TR will
        only be modeled using the corresponding language feature chunk). Also
        note that a larger stimulus window will increase compute time, since it
        increases the amount of stimulus features used to train and test the
        fMRI encoding models.
    movies: list
        List of strings indicating the movies for which the fMRI responses and
        stimulus features are aligned, out of the first six seasons of Friends
        ["friends-s01", "friends-s02", "friends-s03", "friends-s04",
        "friends-s05", "friends-s06"], and the four movies from Movie10
        ["movie10-bourne", "movie10-figures", "movie10-life", "movie10-wolf"].

    Returns
    -------
    aligned_features : float
        Aligned stimulus features for the selected movies.
    aligned_fmri : float
        Aligned fMRI responses for the selected movies.

    """

    ### Empty data variables ###
    aligned_features = []
    aligned_features_summary = []
    aligned_fmri = np.empty((0,1000), dtype=np.float32)
    if all_subject_fmri:
        aligned_fmri = np.empty((0,4,1000), dtype=np.float32)

    ### Loop across movies ###
    for movie in movies:
        #do fmri len check for all mods
        fmri_for_subject = fmri
        if all_subject_fmri:
            fmri_for_subject = fmri['fmri1']
        for mod in features.keys():
            do_features_fmri_len_check(features[mod], fmri_for_subject, movie)
            
        ### Get the IDs of all movies splits for the selected movie ###
        if movie[:7] == 'friends':
            id = movie[8:]
        elif movie[:7] == 'movie10':
            id = movie[8:]
        if not all_subject_fmri:
            movie_splits = [key for key in fmri if id in key[:len(id)]]
        else:
            movie_splits = [key for key in fmri['fmri1'] if id in key[:len(id)]]
        # print('movie[:7]', movie[:7])
        # print('id', id)
        #print('movie_splits', movie_splits)
        #load viewing session
        #print('features.keys()', features['visual'].keys())
        ### Loop over movie splits ###
        for split in movie_splits:
            v_session = None
            if viewing_session is not None:
                v_session, in_session_order_num, max_session_num = viewing_session[split]
                v_session = int(v_session)
                in_session_order_num = int(in_session_order_num)
                max_session_num = int(max_session_num)
                #print('split: ', split, ' v_session: ', v_session, ' in_session_order_num: ', in_session_order_num, ' max_session_num: ', max_session_num)
            # if split == 's01e01a': print('split', split)
            ### Extract the fMRI ###
            if not all_subject_fmri:
                fmri_split = fmri[split]
                # Exclude the first and last fMRI samples
                fmri_split = fmri_split[excluded_samples_start:-excluded_samples_end]
                # print('fmri_split', fmri_split.shape)
                aligned_fmri = np.append(aligned_fmri, fmri_split, 0)
                movie_len = len(fmri_split)
            else:
                fmri1_split = fmri['fmri1'][split]
                fmri2_split = fmri['fmri2'][split]
                fmri3_split = fmri['fmri3'][split]
                fmri5_split = fmri['fmri5'][split]
                # print('fmri1_split', fmri1_split.shape)
                # Exclude the first and last fMRI samples
                fmri1_split = fmri1_split[excluded_samples_start:-excluded_samples_end]
                fmri2_split = fmri2_split[excluded_samples_start:-excluded_samples_end]
                fmri3_split = fmri3_split[excluded_samples_start:-excluded_samples_end]
                fmri5_split = fmri5_split[excluded_samples_start:-excluded_samples_end]
                # print('fmri1_split', fmri1_split.shape)
                # print('fmri2_split', fmri2_split.shape)
                
                # Stack the fMRI data from all subjects into a single array of shape (samples, subjects, features)
                stacked_fmri = np.stack([fmri1_split, fmri2_split, fmri3_split, fmri5_split], axis=1)
                # print('stacked_fmri', stacked_fmri.shape)
                
                aligned_fmri = np.append(aligned_fmri, stacked_fmri, 0)
                # print('aligned_fmri', aligned_fmri.shape)
                movie_len = len(fmri1_split)
            full_split = split
            if split[0] == 's':
                full_split = 'friends_' + split
            range_tupple = (-1,-1)
            # if split == 's01e01a': print('aligned_fmri', aligned_fmri.shape)
            # if split == 's01e01a': print('len(fmri_split', len(fmri_split))
            ### Loop over fMRI samples ###
            for s in range(movie_len):
                # Empty variable containing the stimulus features of all
                # modalities for each fMRI sample
                f_all = np.empty(0)
                ### Loop across modalities ###
                for mod in features.keys():
                    ### Visual and audio features ###
                    # If visual or audio modality, model each fMRI sample using
                    # the N stimulus feature samples up to the fMRI sample of
                    # interest minus the hrf_delay (where N is defined by the
                    # 'stimulus_window' variable)
                    if mod == 'visual' or mod == 'audio' or mod == 'language':
                        # In case there are not N stimulus feature samples up to
                        # the fMRI sample of interest minus the hrf_delay (where
                        # N is defined by the 'stimulus_window' variable), model
                        # the fMRI sample using the first N stimulus feature
                        # samples
                        #print('s', s, 'split', split)
                        effective_split = split
                        # if mod == 'audio':
                        #     effective_split = 'friends_' + split
                        effective_stimulus_window = stimulus_window
                        if mod == 'language':
                            effective_stimulus_window = 2
                        if s < (effective_stimulus_window + hrf_delay):
                            idx_start = excluded_samples_start
                            idx_end = idx_start + effective_stimulus_window
                        else:
                            idx_start = s + excluded_samples_start - hrf_delay \
                                - effective_stimulus_window + 1
                            idx_end = idx_start + effective_stimulus_window
                        # In case there are less visual/audio feature samples
                        # than fMRI samples minus the hrf_delay, use the last N
                        # visual/audio feature samples available (where N is
                        # defined by the 'stimulus_window' variable)
                        if idx_end > (len(features[mod][effective_split])):
                            idx_end = len(features[mod][effective_split])
                            idx_start = idx_end - effective_stimulus_window
                        f = features[mod][effective_split][idx_start:idx_end]
                        range_tupple = (idx_start, idx_end)
                        #print('s', s, 'idx_start', idx_start, 'idx_end', idx_end, mod)
                        f = f.flatten()
                        #print('f', f.shape)
                        # if mod =='visual' and split == 's01e01a' and (s == 0 or s==1 or s==2): 
                        #     print('mod', mod)
                        #     print('s', s)
                        #     print('idx_start', idx_start)
                        #     print('idx_end', idx_end)
                        #     print('f', f.shape)
                        f_all = np.append(f_all, f)


                    ### Language features ###
                    # Since language features already consist of embeddings
                    # spanning several samples, only model each fMRI sample
                    # using the corresponding stimulus feature sample minus the
                    # hrf_delay
                    elif mod == 'language1':
                        # In case there are no language features for the fMRI
                        # sample of interest minus the hrf_delay, model the fMRI
                        # sample using the first language feature sample
                        if s < hrf_delay:
                            idx = excluded_samples_start
                        else:
                            idx = s + excluded_samples_start - hrf_delay
                        # In case there are fewer language feature samples than
                        # fMRI samples minus the hrf_delay, use the last
                        # language feature sample available
                        if idx >= (len(features[mod][split]) - hrf_delay):
                            f = features[mod][split][-1,:]
                        else:
                            f = features[mod][split][idx]
                        f_all = np.append(f_all, f.flatten())
                
                if viewing_session is not None:
                    varr = np.zeros(100)
                    varr[v_session-1] = 1
                    #f_all = np.append(f_all, varr)

                    # 50 features for each 10tr interval, is first, is last
                    varr = np.zeros(54)
                    fr_num = (s//10) 
                    if fr_num > 49:
                        fr_num = 49
                    varr[fr_num] = 1
                    if in_session_order_num == 1:
                        varr[50] = 1
                    if in_session_order_num == max_session_num:
                        varr[51] = 1
                    varr[52] = in_session_order_num /10
                    #print(varr[52])
                    varr[53] = max_session_num /10
                    f_all = np.append(f_all, varr)
                    
                    # 50 features for each 10tr interval
                    varr = np.zeros(50)
                    fr_num = (s//10) 
                    if fr_num > 49:
                        fr_num = 49
                    varr[fr_num] = 1
                    #f_all = np.append(f_all, varr)
                    
                    #sin of frame
                    varr = np.zeros(1)
                    fr_num = (s//10) 
                    if fr_num > 49:
                        fr_num = 49
                    varr[0] = math.sin(normalize_to_radians(fr_num))
                    #f_all = np.append(f_all, varr)
                    
                    # 2 features for start and end
                    varr = np.zeros(2)
                    if s < 10:
                        varr[0] = 1
                    fr_num = (s//10) 
                    if fr_num > 48:
                        varr[1] = 1
                    #f_all = np.append(f_all, varr)

                    # parr = np.zeros(5)
                    # if max_count > 5:
                    #     max_count = 5
                    # #indd = (max_count * 50) + fr_num
                    # parr[max_count-1] = 1
                    #f_all = np.append(f_all, parr)

                    #print('f_all.shape', f_all.shape,'s', s, 'vsession:', str(v_session-1), 'fr_num:', str(fr_num))
                 ### Append the stimulus features of all modalities for this sample ###
                #print('f_all.shape', f_all.shape, 'vsession:', str(v_session-1))
                #print('f_all.shape', f_all.shape)
                aligned_features.append(f_all)
                #print(f_all.shape, movie_len)
                aligned_features_summary.append((full_split, range_tupple))
    ### Convert the aligned features to a numpy array ###
    aligned_features = np.asarray(aligned_features, dtype=np.float32)
    #print('aligned_features.shape', aligned_features.shape)
    ### Output ###
    if summary_features:
        return aligned_features_summary, aligned_fmri
    else:
        return aligned_features, aligned_fmri

def main_feature_extraction():
    root_data_dir = utils.get_data_root_dir()
    subject = 1  #@param ["1", "2", "3", "5"] {type:"raw", allow-input: true}

    modality = "all"  #@param ["visual", "audio", "language", "all"]

    excluded_samples_start = 5  #@param {type:"slider", min:0, max:20, step:1}

    excluded_samples_end = 5  #@param {type:"slider", min:0, max:20, step:1}

    hrf_delay = 3  #@param {type:"slider", min:0, max:10, step:1}

    stimulus_window = 5  #@param {type:"slider", min:1, max:20, step:1}

    movies_train = ["friends-s01", "friends-s02", "friends-s03", "friends-s04", "friends-s05", "movie10-bourne", "movie10-figures", "movie10-life", "movie10-wolf"] # @param {allow-input: true}

    movies_val = ["friends-s06"] # @param {allow-input: true}
    stim_data_dir = utils.get_stimulus_features_dir()
    features = load_stimulus_features(stim_data_dir, modality)

    # Print all available movie splits for each stimulus modality
    for key_modality, value_modality in features.items():
        print(f"\n{key_modality} features movie splits name and shape:")
        for key_movie, value_movie in value_modality.items():
            print(key_movie + " " + str(value_movie.shape))


def save_encoding_accuracy(encoding_accuracy, subject, modality):
    """
    Save encoding accuracy values to a CSV file.
    
    Parameters
    ----------
    encoding_accuracy : numpy.ndarray
        Array containing accuracy values for each parcel
    subject : int
        Subject number
    modality : str
        Feature modality used
    """
    # Create eval_results directory if it doesn't exist
    root_dir = utils.get_data_root_dir()
    eval_dir = os.path.join(root_dir, 'eval_results')
    os.makedirs(eval_dir, exist_ok=True)
    
    # Create DataFrame with index and accuracy columns
    df = pd.DataFrame({
        'index': range(len(encoding_accuracy)),
        'accuracy': encoding_accuracy
    })
    
    # Generate filename
    filename = f'sub-{str(subject).zfill(2)}_modality-{modality}_accuracy.csv'
    filepath = os.path.join(eval_dir, filename)
    
    # Save to CSV
    df.to_csv(filepath, index=False)

def plot_encoding_accuracy(subject, encoding_accuracy, modality):
    print('subject', subject)
    print('modality', modality)
    atlas_file = f'sub-0{subject}_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-dseg_parcellation.nii.gz'
    root_data_dir = utils.get_data_root_dir()
    atlas_path = os.path.join(root_data_dir, 'algonauts_2025.competitors',
        'fmri', f'sub-0{subject}', 'atlas', atlas_file)
    print('atlas_path', atlas_path)
    atlas_masker = NiftiLabelsMasker(labels_img=atlas_path)
    atlas_masker.fit()
    print(encoding_accuracy.shape)
    #encoding_accuracy = np.reshape(encoding_accuracy, (1000, 1))
    print(encoding_accuracy.shape)
    encoding_accuracy_2d = encoding_accuracy.reshape(1, -1)
    encoding_accuracy_nii = atlas_masker.inverse_transform(encoding_accuracy_2d)
    mean_encoding_accuracy = np.round(np.mean(encoding_accuracy), 3)

    ### Plot the encoding accuracy ###
    title = f"Encoding accuracy, sub-0{subject}, modality-{modality}, mean accuracy: " + str(mean_encoding_accuracy)
    display = plotting.plot_glass_brain(
        encoding_accuracy_nii,
        display_mode="lyrz",
        cmap='hot_r',
        colorbar=True,
        plot_abs=False,
        symmetric_cbar=False,
        title=title
    )
    colorbar = display._cbar
    colorbar.set_label("Pearson's $r$", rotation=90, labelpad=12, fontsize=12)
    plotting.show()




def get_model_name(subject, modality, dimension):
    if dimension is not None:
        return f'sub-' + str(subject) + '_modality-' + str(modality) + '_dimension-' + str(dimension)
    else:
        raise ValueError("Dimension is not provided")

def get_features(modality):
    stim_data_dir = utils.get_stimulus_features_dir()
    features = load_stimulus_features(stim_data_dir, modality)
    return features

def get_fmri(subject):
    if utils.isMockMode():
        fmri = {}
        fmri['s01e01a'] = np.random.randn(592, 1000)
        # print('fmri.shape', fmri.keys())
        # print('fmri.shape', fmri['s01e01a'].shape)
        return fmri
    else:
        root_data_dir = utils.get_data_root_dir()
        fmri = load_fmri(root_data_dir, subject)
        return fmri
def add_recurrent_features(features,fmri,recurrence):
    #print('features.shape', features.shape)
    #print('fmri.shape', fmri.shape)
    if recurrence > 0:
        fmri = fmri[recurrence:,:]
        recurrent_features = np.zeros((features.shape[0]-recurrence, features.shape[1]*(recurrence+1)), dtype=np.float32)
        #print('recurrent_features.shape', recurrent_features.shape)
        for i in range(recurrence, features.shape[0]):# Remove first row from fmri_train
            feature_vector = features[i-recurrence:i+1,:].flatten()
            #print('feature_vector.shape', feature_vector.shape)
            recurrent_features[i-recurrence,:] = feature_vector
    else:
        recurrent_features = features
    #print('recurrent_features.shape', recurrent_features.shape)
    #print('fmri.shape', fmri.shape)
    return recurrent_features, fmri

def run_training(features, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train, movies_train_val, training_handler, viewing_session, config):
    #print('run training')
    
    #features_train, fmri_train = add_recurrent_features(features_train, fmri_train, recurrence)
    #features_train_val, fmri_train_val = add_recurrent_features(features_train_val, fmri_train_val, recurrence)
    features_train_val, fmri_train_val = None, None
    if training_handler == 'pytorch':
        features_train, fmri_train = align_features_and_fmri_samples(features, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train, viewing_session)
        trainer = RegressionHander_Pytorch(features_train.shape[1], fmri_train.shape[1])
        print('got simple handler')
    if training_handler == 'loravision':
        all_subjects = False
        if 'fmri1' in fmri.keys():
            all_subjects = True

        print('training for all subjects', all_subjects)
        features_train, fmri_train = align_features_and_fmri_samples(features, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train, viewing_session, summary_features=True, all_subject_fmri=all_subjects)
        output_shape = fmri_train.shape[1]
        if len(fmri_train.shape) == 3:
            output_shape = fmri_train.shape[2]
        #print('feautres_train', features_train[:500])
        print('create trainer')
        del features
        _,_, enable_wandb = utils.get_wandb_config()
        enable_wandb = utils.str_to_bool(enable_wandb)
        print('train enable_wandb', enable_wandb)
        trainer = RegressionHander_Vision(8192 * stimulus_window, output_shape, config['trained_model_name'], enable_wandb=enable_wandb)
        print('got lora vision handler')
        model, training_time = trainer.train(features_train, fmri_train, features_train_val, fmri_train_val, num_gpus=torch.cuda.device_count())
    elif training_handler == 'sklearn':
        #print('aligning features and fmri samples')
        features_train, fmri_train = align_features_and_fmri_samples(features, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train, viewing_session)
        #print('got features and fmri samples')
        trainer = LinearHandler_Sklearn(features_train.shape[1], fmri_train.shape[1])
        model, training_time = trainer.train(features_train, fmri_train, features_train_val, fmri_train_val)
    elif training_handler == 'transformer':
        features_train_val, fmri_train_val = align_features_and_fmri_samples(features, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train_val, viewing_session)
        trainer = RegressionHander_Transformer(features_train.shape[1], fmri_train.shape[1])
        model, training_time = trainer.train(features_train, fmri_train, features_train_val, fmri_train_val, num_gpus=torch.cuda.device_count())
    #print('training')
    
    
    #print('training done')
    del features_train, fmri_train
    return trainer, training_time

def train_for_all_modalities(subject, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train, movies_train_val, training_handler, include_viewing_sessions, config, specific_modalities=None):
    modalities = ["visual", "audio", "language", "all", "audio+language", "visual+language"]
    if specific_modalities:
        modalities = specific_modalities
    viewing_session = None
    if include_viewing_sessions:
        viewing_session = utils.load_viewing_session_for_subject(get_subject_string(subject))
    for modality in modalities:
        print(f"Starting training for modality {modality}...")
        features = get_features(modality)
        print(f"Got features for modality {modality}...")
        trainer, training_time = run_training(features, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train, movies_train_val, training_handler,viewing_session, config)
        print(f"Completed modality {modality} in {training_time:.2f} seconds")
        model_name = get_model_name(subject, modality, stimulus_window)
        trainer.save_model(model_name)
        del features, trainer


def train_for_all_subjects(excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train, movies_train_val, training_handler, include_viewing_sessions, config, specific_modalities=None, skip_if_accuracy_exists=False):
    start_time = time.time()
    for subject in [1, 2, 3, 5]:
        subject_start = time.time()
        print(f"\nStarting training for subject {subject}...")
        if skip_if_accuracy_exists:
            accuracy_json_path = utils.get_accuracy_json_file()
            if does_accuracy_entry_exist(accuracy_json_path, specific_modalities[0], movies_train_val[0], get_subject_string(subject), stimulus_window):
                print(f"Skipping training for subject {subject} as accuracy entry already exists, dimension: {stimulus_window}")
                continue
        fmri = get_fmri(subject)
        #viewing_session = utils.load_viewing_session_for_subject(get_subject_string(subject))
        train_for_all_modalities(subject, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train, movies_train_val, training_handler, include_viewing_sessions, config, specific_modalities)
        subject_time = time.time() - subject_start
        print(f"Completed subject {subject} in {subject_time:.2f} seconds")
        del fmri
    total_time = time.time() - start_time
    print(f"\nTotal training time for all subjects: {total_time:.2f} seconds")
    return total_time

def validate_for_all_modalities(subject, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_val, training_handler, include_viewing_sessions, config, \
                                specific_modalities=None, write_accuracy=False, write_accuracy_to_csv=False, plot_encoding_fig=False, break_up_by_network=False, skip_accuracy_check=False):
    modalities = ["visual", "audio", "language", "all", "audio+language", "visual+language"]
    if specific_modalities:
        modalities = specific_modalities
    for modality in modalities:
        features = get_features(modality)
        accuracy, accuracy_by_network = run_validation(subject, modality, features, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_val, training_handler, include_viewing_sessions, config, write_accuracy, write_accuracy_to_csv=write_accuracy_to_csv, plot_encoding_fig=plot_encoding_fig, break_up_by_network=break_up_by_network, skip_accuracy_check=skip_accuracy_check)
        del features

def validate_for_all_subjects(excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_val, training_handler, include_viewing_sessions, config, specific_modalities=None, write_accuracy=False, write_accuracy_to_csv=False, plot_encoding_fig=False, break_up_by_network=False, save_combined_accuracy=False, experiment_name=None, results_output_directory=None, skip_accuracy_check=False):
    assert len(specific_modalities) == 1
    modality = specific_modalities[0]
    features = get_features(modality)
    
    # Track results for each subject
    subject_accuracies = {}
    subject_accuracies_by_network = {}
    subjects = [1, 2, 3, 5]
    
    for subject in subjects:
        fmri = get_fmri(subject)
        print(f"\nValidation for Subject {subject}")
        accuracy, accuracy_by_network = run_validation(subject, modality, features, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_val, training_handler, include_viewing_sessions, config, write_accuracy, write_accuracy_to_csv=write_accuracy_to_csv, plot_encoding_fig=plot_encoding_fig, break_up_by_network=break_up_by_network, skip_accuracy_check=skip_accuracy_check)
        subject_accuracies[subject] = accuracy
        subject_accuracies_by_network[subject] = accuracy_by_network
        
        del fmri
    del features
    
    # Compute and print average accuracy
    average_accuracy = np.mean(list(subject_accuracies.values()))
    print(f"\n=== SUMMARY RESULTS ===")
    print(f"Individual subject accuracies:")
    for subject in subjects:
        print(f"  Subject {subject}: {subject_accuracies[subject]:.4f}")
    print(f"Average accuracy across all subjects: {average_accuracy:.4f}")
    
    # Compute and print average accuracy by network
    if break_up_by_network and subject_accuracies_by_network[subjects[0]]:
        # Get network names from first subject (assuming all subjects have same networks)
        network_names = [item[0] for item in subject_accuracies_by_network[subjects[0]]]
        
        network_averages = {}
        network_subject_data = {}
        
        for i, network_name in enumerate(network_names):
            network_accuracies = []
            network_subject_data[network_name] = {}
            
            for subject in subjects:
                # Get accuracy for this network (assuming same order for all subjects)
                subject_network_acc = subject_accuracies_by_network[subject][i][1]
                network_accuracies.append(subject_network_acc)
                network_subject_data[network_name][subject] = subject_network_acc
            
            network_averages[network_name] = np.mean(network_accuracies)
        
        print(f"\nAverage accuracy by network:")
        for network, avg_acc in network_averages.items():
            print(f"  {network}: {avg_acc:.4f}")
        
        # Save to CSV if requested
        if save_combined_accuracy:
            import pandas as pd
            import os
            
            # Create data for CSV
            csv_data = []
            for network_name in network_names:
                row = {'network': network_name, 'average': network_averages[network_name]}
                
                # Add individual subject data
                for subject in subjects:
                    column_name = f'subject {subject}'
                    row[column_name] = network_subject_data[network_name][subject]
                
                csv_data.append(row)
            
            # Create DataFrame and save
            df = pd.DataFrame(csv_data)

            filepath = utils.get_subject_network_accuracy_file_for_experiement(experiment_name, results_output_directory)
            
            df.to_csv(filepath, index=False)
            print(f"\nCombined network accuracy saved to: {filepath}")
    
    print("=== END SUMMARY ===\n")

def run_validation_by_average(subject, modality, fmri,excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train, movies_val,training_handler, include_viewing_sessions, write_accuracy=False, write_accuracy_to_csv=False, plot_encoding_fig=False,break_up_by_network=False):
    viewing_session = None
    if include_viewing_sessions:
        viewing_session = utils.load_viewing_session_for_subject(get_subject_string(subject))
    features = get_features(modality)
    features_val, fmri_val = align_features_and_fmri_samples(features, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_val, viewing_session)
    features_train, fmri_train = align_features_and_fmri_samples(features, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train, viewing_session)
    print('features_train.shape', features_train.shape)
    print('features_val.shape', features_val.shape)
    print('fmri_train.shape', fmri_train.shape)
    print('fmri_val.shape', fmri_val.shape)
    #get average of fmri_train
    fmri_train_avg = np.mean(fmri_train, axis=0)
    print('fmri_train_avg.shape', fmri_train_avg.shape)

    fmri_val2 = fmri_val[1:,:]
    fmri_val3 = fmri_val[:22852,:]
    print('fmri_val2.shape', fmri_val2.shape)
    print('fmri_val3.shape', fmri_val3.shape)
    # Expand fmri_train_avg to match fmri_val shape
    # Create array with shape (n_samples, n_voxels) where each row is the average
    fmri_val_pred = np.tile(fmri_train_avg, (fmri_val.shape[0], 1))
    noise = np.random.normal(0, 0.0001, size=fmri_val_pred.shape)
    fmri_val_pred = fmri_val_pred + noise
    print('fmri_val_pred.shape', fmri_val_pred.shape)
    print('fmri_val.shape', fmri_val.shape)
    print(fmri_val_pred[0,100], fmri_val_pred[1,100])
    print(fmri_val[0,100], fmri_val[1,100])
    print(fmri_val_pred[1,999], fmri_val_pred[519,999])
    print(fmri_val[1,999], fmri_val[519,999])
    #run validation
    if break_up_by_network:
        prediction_by_network = get_breakup_by_network(fmri_val2, fmri_val3)
        accuracy_by_network = []
        for prediction in prediction_by_network:
            measure, network_fmri_val, network_fmri_val_pred = prediction
            accuracy, encoding_accuracy = utils.compute_encoding_accuracy(network_fmri_val, network_fmri_val_pred, subject, measure, print_output=False, write_to_csv=write_accuracy_to_csv)
            print(measure, 'accuracy', accuracy)
            accuracy_by_network.append((measure, accuracy))
        json_path = utils.get_network_accuracy_json_file()
        utils.append_network_accuracies_to_json(json_path, accuracy_by_network)
    else:
        accuracy, encoding_accuracy = utils.compute_encoding_accuracy(fmri_val2, fmri_val3, subject, modality)
    return accuracy


def run_validation(subject, modality, features, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_val,training_handler, include_viewing_sessions, config, \
    write_accuracy=False, write_accuracy_to_csv=False, plot_encoding_fig=False,break_up_by_network=False, skip_accuracy_check=False):
    viewing_session = None
    if include_viewing_sessions:
        viewing_session = utils.load_viewing_session_for_subject(get_subject_string(subject))
   
    #features_val, fmri_val = add_recurrent_features(features_val, fmri_val, recurrence)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # features_val = torch.FloatTensor(features_val).to(device)
    fmri_val = None
    fmri_val_pred = []
    if training_handler == 'pytorch':
         # Align the stimulus features with the fMRI responses for the validation movies
        features_val, fmri_val = align_features_and_fmri_samples(features, fmri,
        excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window,
        movies_val, viewing_session)
        trainer = RegressionHander_Pytorch(features_val.shape[1], fmri_val.shape[1])
    elif training_handler == 'sklearn':
         # Align the stimulus features with the fMRI responses for the validation movies
        if movies_val[0] == 'friends-s07':
            fmri, boundary = prepare_s7_fmri_for_alignment(subject)
            skip_accuracy_check = True
        features_val, fmri_val = align_features_and_fmri_samples(features, fmri,
        excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window,
        movies_val, viewing_session)
        trainer = LinearHandler_Sklearn(features_val.shape[1], fmri_val.shape[1])
    elif training_handler == 'transformer':
         # Align the stimulus features with the fMRI responses for the validation movies
        features_val, fmri_val = align_features_and_fmri_samples(features, fmri,
        excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window,
        movies_val, viewing_session)
        trainer = RegressionHander_Transformer(features_val.shape[1], fmri_val.shape[1])
    elif training_handler == 'loravision':
        assert len(movies_val) == 1, "loravision only supports one movie for validation"
        if movies_val[0] == 'friends-s07':
            fmri, boundary = prepare_s7_fmri_for_alignment(subject)
            skip_accuracy_check = True
        boundary = get_boundary_from_fmri_for_movie_for_subject(subject, movies_val[0])
        features_val, fmri_val = align_features_and_fmri_samples(features, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_val, viewing_session, summary_features=True, all_subject_fmri=False)
        del features
        _,_, enable_wandb = utils.get_wandb_config()
        lora_model = utils.get_model_checkpoint()
        trainer = RegressionHander_Vision(8192 * stimulus_window, fmri_val.shape[1], pretrain_params_name=lora_model, enable_wandb=False)
        assert len(features_val) == fmri_val.shape[0], f"features_val.shape[0] {features_val.shape[0]} != fmri_val.shape[0] {fmri_val.shape[0]}"
        from_idx = 0
        total_size =0
        num_stimuli =0
        for stim_id, size in boundary:
            num_stimuli +=1
            total_size += size
            effective_size = size-10
            features_val_stim = features_val[from_idx:from_idx+effective_size]
            fmri_val_stim = fmri_val[from_idx:from_idx+effective_size,:]
            if stim_id.startswith('s'):
                prefix= "friends_" + stim_id
            else:
                prefix = "movie10_" + stim_id
            fmri_val_pred_stim = trainer.predict(features_val_stim, prefix)
            fmri_val_pred.append(fmri_val_pred_stim)
            #prefix with stim_id
            
            
            from_idx = from_idx + effective_size
            assert (len(features_val_stim) + 10) == size, f"size mismatch while slicing {stim_id} {len(features_val_stim) + 10} {size}"
            assert (fmri_val_stim.shape[0] + 10) == size, f"size mismatch while slicing {stim_id} {fmri_val_stim.shape[0] + 10} {size}"
        assert total_size == (len(features_val) + num_stimuli*10), f"total_size {total_size} != features_val.shape[0] {len(features_val) + num_stimuli*10}"
        assert total_size == (fmri_val.shape[0] + num_stimuli*10), f"total_size {total_size} != fmri_val.shape[0] {fmri_val.shape[0] + num_stimuli*10}"
        fmri_val_pred = np.concatenate(fmri_val_pred, axis=0)

    if training_handler != 'loravision':
        if not config['trained_model_name']:
            model_name = get_model_name(subject, modality, stimulus_window)
        else:
            model_name = config['trained_model_name']
        print('model_name', model_name)
        trainer.load_model(model_name)

    if training_handler != 'loravision':
        fmri_val_pred = trainer.predict(features_val)
    #save it first
    movie_name = None
    if len(movies_val) == 1:
        movie_name = movies_val[0]
    utils.save_predictions_accuracy(subject, movie_name, fmri_val_pred, None)
    #print('fmri_val_pred.shape', fmri_val_pred.shape)
    full_accuracy = 0
    full_encoding_accuracy = None
    accuracy_by_network = []
    if not skip_accuracy_check:       
        if break_up_by_network:
            prediction_by_network = get_breakup_by_network(fmri_val, fmri_val_pred)
            for prediction in prediction_by_network:
                measure, network_fmri_val, network_fmri_val_pred = prediction
                network_accuracy, network_encoding_accuracy = utils.compute_encoding_accuracy(network_fmri_val, network_fmri_val_pred, subject, measure, print_output=False, write_to_csv=write_accuracy_to_csv)
                print(measure, 'accuracy', network_accuracy)
                accuracy_by_network.append((measure, network_accuracy))
            json_path = utils.get_network_accuracy_json_file()
            utils.append_network_accuracies_to_json(json_path, accuracy_by_network)
        # else:
        full_accuracy, full_encoding_accuracy = utils.compute_encoding_accuracy(fmri_val, fmri_val_pred, subject, modality, write_to_csv=write_accuracy_to_csv)
        #print('encoding_accuracy.shape', full_encoding_accuracy.shape)
        utils.save_predictions_accuracy(subject, movie_name, None, full_encoding_accuracy)
        if plot_encoding_fig:
            #encoding_accuracy = np.zeros((1000,), dtype=np.float32)
            # encoding_accuracy[:] = 0
            # encoding_accuracy[173:232] = 1
            # encoding_accuracy[684:744] = 1
            print('encoding_accuracy.shape', full_encoding_accuracy.shape)
            plot_encoding_accuracy(subject, full_encoding_accuracy, modality)
        if write_accuracy:
            acc_json_path = utils.get_accuracy_json_file()
            update_accuracy_json(acc_json_path, float(np.mean(full_accuracy)), modality, movies_val[0], subject, stimulus_window)
    
    return full_accuracy, accuracy_by_network

def get_subject_string(subject):
    if subject == 1:
        return 'sub-01'
    elif subject == 2:
        return 'sub-02'
    elif subject == 3:
        return 'sub-03'
    elif subject == 5:
        return 'sub-05'

def measure_yony_accuracy(subject, modality, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train):
    fmri = get_fmri(subject)
    viewing_session = utils.load_viewing_session_for_subject(get_subject_string(subject))
    features = get_features(modality)
    features_train, fmri_train = align_features_and_fmri_samples(features, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train, viewing_session)
    
    #utils.compute_encoding_accuracy(fmri_train, fmri_train, subject, modality)
    pre_fmri = fmri_train.copy()
    # Remove first row from fmri_train
    fmri_train = fmri_train[1:, :]  # Start from index 1 to end
    # Remove last row from pre_fmri
    pre_fmri = pre_fmri[:-1, :] 
    print('fmri_train.shape', fmri_train.shape)
    print('pre_fmri.shape', pre_fmri.shape)
    utils.compute_encoding_accuracy(fmri_train, pre_fmri, subject, modality)

def does_accuracy_entry_exist(json_path, modality, val_video, subject, dimension):
    if not os.path.exists(json_path):
        return False
    with open(json_path, 'r') as f:
        data = json.load(f)
    for item in data:
        if item['modality'] == modality and item['val_video'] == val_video and item['subject'] == subject and item['dimension'] == dimension:
            return True
    return False

def update_accuracy_json(json_path, accuracy, modality, val_video, subject, dimension):
    """
    Updates a JSON file with accuracy information for a specific modality, validation video, and subject.
    If an entry already exists for the given combination, it updates the accuracy.
    Otherwise, it creates a new entry.
    
    Parameters
    ----------
    json_path : str
        Path to the JSON file to update
    accuracy : float
        Accuracy value to store
    modality : str
        Feature modality used (e.g., "visual", "audio", "language", "all")
    val_video : str
        Validation video name (e.g., "friends-s06")
    subject : int or str
        Subject number or identifier
    dimension : str or int, optional
        Additional dimension to categorize the entry (e.g., "cortical", "subcortical", or a specific ROI number)
    """
    # Get current timestamp
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Convert subject to string format if it's an integer
    if isinstance(subject, int):
        subject = get_subject_string(subject)
    
    # Create the entry dictionary
    entry = {
        "modality": modality,
        "val_video": val_video,
        "subject": subject,
        "accuracy": float(accuracy),
        "dimension": dimension,
        "updated_on": current_time
    }
    
    # Load existing data if file exists, otherwise create empty list
    data = []
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            # If file exists but is not valid JSON, start with empty list
            data = []
    
    # Check if entry already exists
    entry_exists = False
    for i, item in enumerate(data):
        if (item.get("modality") == modality and 
            item.get("val_video") == val_video and 
            item.get("subject") == subject and 
            item.get("dimension") == dimension):
            # Update existing entry
            data[i]["accuracy"] = entry["accuracy"]
            data[i]["updated_on"] = current_time
            entry_exists = True
            break
    
    operation = 'Updated'
    # If entry doesn't exist, add it
    if not entry_exists:
        data.append(entry)
        operation = 'Added'

    # Write updated data back to file
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    
    print(f"{operation} {json_path} with accuracy {accuracy} for {subject}, {modality}, {val_video}, {dimension} at {current_time}")

def plot_accuracy_by_dimension(json_path, subject=None, modality=None, val_video=None, figsize=(12, 6), 
                              save_path=None, show_average=True, y_min=0.13, y_max=0.25):
    """
    Plot accuracy values from a JSON file with dimension on the x-axis and accuracy on the y-axis.
    
    Parameters
    ----------
    json_path : str
        Path to the JSON file containing accuracy data
    subject : str or list, optional
        Filter by specific subject(s) (e.g., 'sub-01' or ['sub-01', 'sub-02'])
    modality : str or list, optional
        Filter by specific modality/modalities (e.g., 'visual' or ['visual', 'audio'])
    val_video : str or list, optional
        Filter by specific validation video(s) (e.g., 'friends-s06')
    figsize : tuple, optional
        Figure size as (width, height)
    save_path : str, optional
        Path to save the figure. If None, the figure is displayed but not saved.
    show_average : bool, optional
        If True, adds a plot showing the average across all subjects
    y_min : float, optional
        Minimum value for y-axis (default: 0.13)
    y_max : float, optional
        Maximum value for y-axis (default: 0.25)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object
    """
    import json
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    # Load the JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(data)
    
    # Apply filters for modality and val_video
    if modality:
        if isinstance(modality, str):
            modality = [modality]
        df = df[df['modality'].isin(modality)]
    
    if val_video:
        if isinstance(val_video, str):
            val_video = [val_video]
        df = df[df['val_video'].isin(val_video)]
    
    # Make a copy of the full filtered dataframe before applying subject filter
    df_all = df.copy()
    
    # Apply subject filter if provided
    if subject:
        if isinstance(subject, str):
            subject = [subject]
        df = df[df['subject'].isin(subject)]
    
    # Check if we have data after filtering
    if df.empty:
        print("No data available after applying filters.")
        return None
    
    # Convert dimension to numeric if possible for proper sorting
    try:
        df['dimension'] = pd.to_numeric(df['dimension'])
        df = df.sort_values('dimension')
        df_all['dimension'] = pd.to_numeric(df_all['dimension'])
        df_all = df_all.sort_values('dimension')
    except:
        # If conversion fails, keep as is (might be string dimensions)
        pass
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # If we have multiple subjects, modalities, or val_videos, use different colors/markers
    groupby_columns = []
    if len(df['subject'].unique()) > 1 and subject is None:
        groupby_columns.append('subject')
    if len(df['modality'].unique()) > 1 and modality is None:
        groupby_columns.append('modality')
    if len(df['val_video'].unique()) > 1 and val_video is None:
        groupby_columns.append('val_video')
    
    # If we have grouping columns, plot each group separately
    if groupby_columns:
        for name, group in df.groupby(groupby_columns):
            label = ' + '.join([str(n) for n in name]) if isinstance(name, tuple) else str(name)
            ax.plot(group['dimension'], group['accuracy'], 'o-', alpha=0.7, linewidth=1.5, label=label)
    else:
        # Otherwise, plot all data points with the same style
        ax.plot(df['dimension'], df['accuracy'], 'o-', alpha=0.7, linewidth=1.5, label='Individual')
    
    # Add average across subjects if requested
    if show_average and 'subject' in df_all.columns and len(df_all['subject'].unique()) > 1:
        # Group by dimension and calculate mean accuracy
        avg_df = df_all.groupby('dimension')['accuracy'].mean().reset_index()
        # Plot average with thicker line and distinct color
        ax.plot(avg_df['dimension'], avg_df['accuracy'], 'o-', color='red', 
                linewidth=2.5, markersize=8, label='Average across subjects')
        
        # Add standard deviation as shaded area if we have multiple subjects
        if len(df_all['subject'].unique()) > 2:
            std_df = df_all.groupby('dimension')['accuracy'].std().reset_index()
            ax.fill_between(avg_df['dimension'], 
                           avg_df['accuracy'] - std_df['accuracy'],
                           avg_df['accuracy'] + std_df['accuracy'],
                           alpha=0.2, color='red')
    
    # Set y-axis limits
    ax.set_ylim(y_min, y_max)
    
    # Add a horizontal line at y=0 for reference (only if visible in the y-range)
    if y_min <= 0 <= y_max:
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Set labels and title
    ax.set_xlabel('HRF Delay')
    ax.set_ylabel('Accuracy (Pearson\'s r)')
    
    # Create title based on filters
    title_parts = []
    if subject:
        title_parts.append(f"Subject: {', '.join(subject)}")
    if modality:
        title_parts.append(f"Modality: {', '.join(modality)}")
    if val_video:
        title_parts.append(f"Validation: {', '.join(val_video)}")
    
    title = "Accuracy by HRF Delay" + (" - " + " | ".join(title_parts) if title_parts else "")
    ax.set_title(title)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    # Add mean accuracy as text annotation
    mean_acc = df['accuracy'].mean()
    ax.text(0.02, 0.95, f'Mean Accuracy: {mean_acc:.4f}', 
            transform=ax.transAxes, fontsize=10, 
            bbox=dict(facecolor='white', alpha=0.7))
    
    # Add legend
    ax.legend(loc='best')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


