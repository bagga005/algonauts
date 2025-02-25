import numpy as np
import torch
import torch.nn as nn
import h5py
import os
import utils
from model_sklearn import LinearHandler_Sklearn
from model_torchregression import RegressionHander_Pytorch
from model_transformer import RegressionHander_Transformer
import nibabel as nib
from nilearn import plotting
from nilearn.maskers import NiftiLabelsMasker
import time
import pandas as pd
import matplotlib.pyplot as plt
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
    if modality == 'visual' or modality == 'all' or modality == 'visual+language':
        stimuli_dir = os.path.join(root_data_dir, 'pca',
            'friends_movie10', 'visual', 'features_train.npy')
        features['visual'] = np.load(stimuli_dir, allow_pickle=True).item()

    ### Load the audio features ###
    if modality == 'audio' or modality == 'all' or modality == 'audio+language':
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
        del fmri[keys_movie[0]]
        del fmri[keys_movie[1]]
    # Average the fMRI responses across the two repeats for 'life'
    keys_all = fmri.keys()
    life_splits = 5
    for s in range(life_splits):
        movie = 'life' + format(s+1, '02')
        keys_movie = [rep for rep in keys_all if movie in rep]
        fmri[movie] = ((fmri[keys_movie[0]] + fmri[keys_movie[1]]) / 2).astype(np.float32)
        del fmri[keys_movie[0]]
        del fmri[keys_movie[1]]

    ### Output ###
    return fmri



def align_features_and_fmri_samples(features, fmri, excluded_samples_start,
    excluded_samples_end, hrf_delay, stimulus_window, movies, viewing_session):
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
        the fMRI samples (i.e., the fMRI TRs), since in some cases the fMRI run
        ran longer than the actual movie. However, keep in mind that the fMRI
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
    aligned_fmri = np.empty((0,1000), dtype=np.float32)

    ### Loop across movies ###
    for movie in movies:

        ### Get the IDs of all movies splits for the selected movie ###
        if movie[:7] == 'friends':
            id = movie[8:]
        elif movie[:7] == 'movie10':
            id = movie[8:]
        movie_splits = [key for key in fmri if id in key[:len(id)]]
        # print('movie[:7]', movie[:7])
        # print('id', id)
        # print('movie_splits', movie_splits)
        #load viewing session
        
        ### Loop over movie splits ###
        for split in movie_splits:
            v_session = None
            if viewing_session is not None:
                v_session, max_count = viewing_session[split]
                v_session = int(v_session)
                max_count = int(max_count)
                #print('split: ', split, ' v_session: ', v_session, ' max_count: ', max_count)
            # if split == 's01e01a': print('split', split)
            ### Extract the fMRI ###
            fmri_split = fmri[split]
            # if split == 's01e01a': print('fmri_split', fmri_split.shape)
            # Exclude the first and last fMRI samples
            fmri_split = fmri_split[excluded_samples_start:-excluded_samples_end]
            # if split == 's01e01a': print('fmri_split', fmri_split.shape)
            aligned_fmri = np.append(aligned_fmri, fmri_split, 0)
            # if split == 's01e01a': print('aligned_fmri', aligned_fmri.shape)
            # if split == 's01e01a': print('len(fmri_split', len(fmri_split))
            ### Loop over fMRI samples ###
            for s in range(len(fmri_split)):
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
                    if mod == 'visual' or mod == 'audio':
                        # In case there are not N stimulus feature samples up to
                        # the fMRI sample of interest minus the hrf_delay (where
                        # N is defined by the 'stimulus_window' variable), model
                        # the fMRI sample using the first N stimulus feature
                        # samples
                        #print('s', s, 'split', split)
                        if s < (stimulus_window + hrf_delay):
                            idx_start = excluded_samples_start
                            idx_end = idx_start + stimulus_window
                        else:
                            idx_start = s + excluded_samples_start - hrf_delay \
                                - stimulus_window + 1
                            idx_end = idx_start + stimulus_window
                        # In case there are less visual/audio feature samples
                        # than fMRI samples minus the hrf_delay, use the last N
                        # visual/audio feature samples available (where N is
                        # defined by the 'stimulus_window' variable)
                        if idx_end > (len(features[mod][split])):
                            idx_end = len(features[mod][split])
                            idx_start = idx_end - stimulus_window
                        f = features[mod][split][idx_start:idx_end]
                        # if mod =='visual' and split == 's01e01a' and (s == 0 or s==1 or s==2): 
                        #     print('mod', mod)
                        #     print('s', s)
                        #     print('idx_start', idx_start)
                        #     print('idx_end', idx_end)
                        #     print('f', f.shape)
                        f_all = np.append(f_all, f.flatten())

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
                    varr = np.zeros(50)
                    fr_num = (s//10) 
                    if fr_num > 49:
                        fr_num = 49
                    varr[fr_num] = 1
                    #f_all = np.append(f_all, varr)
                    parr = np.zeros(5)
                    if max_count > 5:
                        max_count = 5
                    #indd = (max_count * 50) + fr_num
                    parr[max_count-1] = 1
                    f_all = np.append(f_all, parr)

                    #print('f_all.shape', f_all.shape,'s', s, 'vsession:', str(v_session-1), 'fr_num:', str(fr_num))
                 ### Append the stimulus features of all modalities for this sample ###
                #print('f_all.shape', f_all.shape, 'vsession:', str(v_session-1))
                aligned_features.append(f_all)

    ### Convert the aligned features to a numpy array ###
    aligned_features = np.asarray(aligned_features, dtype=np.float32)

    ### Output ###
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
    encoding_accuracy_nii = atlas_masker.inverse_transform(encoding_accuracy)
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




def get_model_name(subject, modality):
    return f'sub-' + str(subject) + '_modality-' + str(modality)

def get_features(modality):
    stim_data_dir = utils.get_stimulus_features_dir()
    features = load_stimulus_features(stim_data_dir, modality)
    return features

def get_fmri(subject):
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

def run_training(features, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train, movies_train_val, training_handler, viewing_session, recurrence=1):
    features_train, fmri_train = align_features_and_fmri_samples(features, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train, viewing_session)
    #features_train, fmri_train = add_recurrent_features(features_train, fmri_train, recurrence)
    #features_train_val, fmri_train_val = add_recurrent_features(features_train_val, fmri_train_val, recurrence)
    features_train_val, fmri_train_val = None, None
    if training_handler == 'pytorch':
        trainer = RegressionHander_Pytorch(features_train.shape[1], fmri_train.shape[1])
    elif training_handler == 'sklearn':
        trainer = LinearHandler_Sklearn(features_train.shape[1], fmri_train.shape[1])
    elif training_handler == 'transformer':
        features_train_val, fmri_train_val = align_features_and_fmri_samples(features, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train_val, viewing_session)
        trainer = RegressionHander_Transformer(features_train.shape[1], fmri_train.shape[1])
    model, training_time = trainer.train(features_train, fmri_train, features_train_val, fmri_train_val)
    del features_train, fmri_train
    return trainer, training_time

def train_for_all_modalities(subject, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train, movies_train_val, training_handler, include_viewing_sessions, specific_modalities=None, recurrence=1):
    modalities = ["visual", "audio", "language", "all", "audio+language", "visual+language"]
    if specific_modalities:
        modalities = specific_modalities
    viewing_session = None
    if include_viewing_sessions:
        viewing_session = utils.load_viewing_session_for_subject(get_subject_string(subject))
    for modality in modalities:
        print(f"Starting training for modality {modality}...")
        features = get_features(modality)
        trainer, training_time = run_training(features, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train, movies_train_val, training_handler,viewing_session, recurrence=recurrence)
        print(f"Completed modality {modality} in {training_time:.2f} seconds")
        model_name = get_model_name(subject, modality)
        trainer.save_model(model_name)
        del features


def train_for_all_subjects(excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train, movies_train_val, training_handler, include_viewing_sessions, specific_modalities=None, recurrence=1):
    start_time = time.time()
    for subject in [1, 2, 3, 5]:
        subject_start = time.time()
        print(f"Starting training for subject {subject}...")
        fmri = get_fmri(subject)
        #viewing_session = utils.load_viewing_session_for_subject(get_subject_string(subject))
        train_for_all_modalities(subject, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train, movies_train_val, training_handler, include_viewing_sessions, specific_modalities, recurrence)
        subject_time = time.time() - subject_start
        print(f"Completed subject {subject} in {subject_time:.2f} seconds")
        del fmri
    total_time = time.time() - start_time
    print(f"\nTotal training time for all subjects: {total_time:.2f} seconds")
    return total_time

def validate_for_all_modalities(subject, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_val, training_handler, include_viewing_sessions, specific_modalities=None, recurrence=1):
    modalities = ["visual", "audio", "language", "all", "audio+language", "visual+language"]
    if specific_modalities:
        modalities = specific_modalities
    for modality in modalities:
        features = get_features(modality)
        run_validation(subject, modality, features, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_val, training_handler, include_viewing_sessions, recurrence)
        del features

def validate_for_all_subjects(excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_val, training_handler, include_viewing_sessions, specific_modalities=None, recurrence=1):
    for subject in [1, 2, 3, 5]:
        fmri = get_fmri(subject)
        validate_for_all_modalities(subject, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_val, training_handler, include_viewing_sessions, specific_modalities, recurrence)
        del fmri

def run_validation(subject, modality, features, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_val,training_handler, include_viewing_sessions, recurrence=1):
    viewing_session = None
    if include_viewing_sessions:
        viewing_session = utils.load_viewing_session_for_subject(get_subject_string(subject))
    # Align the stimulus features with the fMRI responses for the validation movies
    features_val, fmri_val = align_features_and_fmri_samples(features, fmri,
        excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window,
        movies_val, viewing_session)
    #features_val, fmri_val = add_recurrent_features(features_val, fmri_val, recurrence)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # features_val = torch.FloatTensor(features_val).to(device)
    
    if training_handler == 'pytorch':
        trainer = RegressionHander_Pytorch(features_val.shape[1], fmri_val.shape[1])
    elif training_handler == 'sklearn':
        trainer = LinearHandler_Sklearn(features_val.shape[1], fmri_val.shape[1])
    elif training_handler == 'transformer':
        trainer = RegressionHander_Transformer(features_val.shape[1], fmri_val.shape[1])

    model_name = get_model_name(subject, modality)
    trainer.load_model(model_name)

    fmri_val_pred = trainer.predict(features_val)
    
    utils.compute_encoding_accuracy(fmri_val, fmri_val_pred, subject, modality)

def get_subject_string(subject):
    if subject == 1:
        return '01'
    elif subject == 2:
        return '02'
    elif subject == 3:
        return '03'
    elif subject == 5:
        return '05'
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


