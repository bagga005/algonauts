import numpy as np
import h5py
import os
import utils
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
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
        stimuli_dir = os.path.join(root_data_dir, 'stimulus_features', 'pca',
            'friends_movie10', 'visual', 'features_train.npy')
        features['visual'] = np.load(stimuli_dir, allow_pickle=True).item()

    ### Load the audio features ###
    if modality == 'audio' or modality == 'all' or modality == 'audio+language':
        stimuli_dir = os.path.join(root_data_dir, 'stimulus_features', 'pca',
            'friends_movie10', 'audio', 'features_train.npy')
        features['audio'] = np.load(stimuli_dir, allow_pickle=True).item()

    ### Load the language features ###
    if modality == 'language' or modality == 'all' or modality == 'audio+language' or modality == 'visual+language':
        stimuli_dir = os.path.join(root_data_dir, 'stimulus_features', 'pca',
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
    excluded_samples_end, hrf_delay, stimulus_window, movies):
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

        ### Loop over movie splits ###
        for split in movie_splits:

            ### Extract the fMRI ###
            fmri_split = fmri[split]
            # Exclude the first and last fMRI samples
            fmri_split = fmri_split[excluded_samples_start:-excluded_samples_end]
            aligned_fmri = np.append(aligned_fmri, fmri_split, 0)

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

                 ### Append the stimulus features of all modalities for this sample ###
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
    features = load_stimulus_features(root_data_dir, modality)

    # Print all available movie splits for each stimulus modality
    for key_modality, value_modality in features.items():
        print(f"\n{key_modality} features movie splits name and shape:")
        for key_movie, value_movie in value_modality.items():
            print(key_movie + " " + str(value_movie.shape))

def train_encoding(features_train, fmri_train):
    """
    Train a linear-regression-based encoding model to predict fMRI responses
    using movie features.

    Parameters
    ----------
    features_train : float
        Stimulus features for the training movies.
    fmri_train : float
        fMRI responses for the training movies.

    Returns
    -------
    model : object
        Trained regression model.
    training_time : float
        Time taken to train the model in seconds.
    """
    
    ### Record start time ###
    start_time = time.time()
    
    ### Train the linear regression model ###
    model = LinearRegression().fit(features_train, fmri_train)
    
    ### Calculate training time ###
    training_time = time.time() - start_time

    ### Output ###
    return model, training_time

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

def compute_encoding_accuracy(fmri_val, fmri_val_pred, subject, modality):
    """
    Compare the  recorded (ground truth) and predicted fMRI responses, using a
    Pearson's correlation. The comparison is perfomed independently for each
    fMRI parcel. The correlation results are then plotted on a glass brain.

    Parameters
    ----------
    fmri_val : float
        fMRI responses for the validation movies.
    fmri_val_pred : float
        Predicted fMRI responses for the validation movies
    subject : int
        Subject number used to train and validate the encoding model.
    modality : str
        Feature modality used to train and validate the encoding model.

    """

    ### Correlate recorded and predicted fMRI responses ###
    encoding_accuracy = np.zeros((fmri_val.shape[1]), dtype=np.float32)
    for p in range(len(encoding_accuracy)):
        encoding_accuracy[p] = pearsonr(fmri_val[:, p],
            fmri_val_pred[:, p])[0]
    print('encoding_accuracy.shape', encoding_accuracy.shape)
    mean_encoding_accuracy = np.round(np.mean(encoding_accuracy), 3)
    std_encoding_accuracy = np.round(np.std(encoding_accuracy), 3)
    print(f"Encoding accuracy, sub-0{subject}, modality-{modality}, mean accuracy: {mean_encoding_accuracy}, std: {std_encoding_accuracy}")

    #plot_encoding_accuracy(subject, encoding_accuracy, modality)
    utils.save_npy(encoding_accuracy, subject, modality)
    # Save accuracy values to CSV
    save_encoding_accuracy(encoding_accuracy, subject, modality)


def get_model_name(subject, modality):
    return f'sub-' + str(subject) + '_modality-' + str(modality)

def get_features(modality):
    root_data_dir = utils.get_data_root_dir()
    features = load_stimulus_features(root_data_dir, modality)
    return features

def get_fmri(subject):
    root_data_dir = utils.get_data_root_dir()
    fmri = load_fmri(root_data_dir, subject)
    return fmri

def run_training(features, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train):
    features_train, fmri_train = align_features_and_fmri_samples(features, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train)
    model, training_time = train_encoding(features_train, fmri_train)
    del features_train, fmri_train
    return model, training_time

def train_for_all_modalities(subject, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train):
    for modality in ["visual", "audio", "language", "all", "audio+language", "visual+language"]:
        features = get_features(modality)
        model, training_time = run_training(features, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train)
        model_name = get_model_name(subject, modality)
        utils.save_model(model, model_name)
        del features

def train_for_all_subjects(excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train):
    for subject in [1, 2, 3, 5]:
        fmri = get_fmri(subject)
        train_for_all_modalities(subject, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train)
        del fmri

def validate_for_all_modalities(subject, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_val):
    for modality in ["visual", "audio", "language", "all", "audio+language", "visual+language"]:
        features = get_features(modality)
        run_validation(subject, modality, features, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_val)
        del features

def validate_for_all_subjects(excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_val):
    for subject in [1, 2, 3, 5]:
        fmri = get_fmri(subject)
        validate_for_all_modalities(subject, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_val)
        del fmri

def run_validation(subject, modality, features, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_val):
 # Align the stimulus features with the fMRI responses for the validation movies
    features_val, fmri_val = align_features_and_fmri_samples(features, fmri,
        excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window,
        movies_val)
    model_name = get_model_name(subject, modality)
    model = utils.load_model(model_name)

    # Predict the fMRI responses for the validation movies
    fmri_val_pred = model.predict(features_val)
    compute_encoding_accuracy(fmri_val, fmri_val_pred, subject, modality)

def main():
    root_data_dir = utils.get_data_root_dir()
    excluded_samples_start = 5  #@param {type:"slider", min:0, max:20, step:1}
    excluded_samples_end = 5  #@param {type:"slider", min:0, max:20, step:1}
    hrf_delay = 3  #@param {type:"slider", min:0, max:10, step:1}
    stimulus_window = 5  #@param {type:"slider", min:1, max:20, step:1}
    movies_train = ["friends-s01", "friends-s02", "friends-s03", "friends-s04", "friends-s05", "movie10-bourne", "movie10-figures", "movie10-life", "movie10-wolf"] # @param {allow-input: true}
    movies_val = ["friends-s06"] # @param {allow-input: true}
    #movies_train = ["friends-s01"] # @param {allow-input: true}

    # modality = "all"
    # features = get_features(modality)
    # #print('features.keys()', features.keys())
    # subject = 2
    # fmri = get_fmri(subject)
    # areas_of_interest_path = os.path.join(root_data_dir, 'eval_results', 'areas-of-interest.csv')
    # arr = utils.load_csv_to_array(areas_of_interest_path)
    # print('arr', arr[:100])
    # print('arr.shape', arr.shape)
    # utils.save_npy(arr, subject, modality)
    #features_train, fmri_train = align_features_and_fmri_samples(features, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train)
    #run_validation(subject, modality, features, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_val)
    # print('modality', modality)
    # print('features_train.shape', features_train.shape)
    # print('fmri_train.shape', fmri_train.shape)
    train_for_all_subjects(excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train)
    #train_for_all_modalities(subject, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train)
    #validate_for_all_modalities(subject, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_val)
    #validate_for_all_subjects(excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_val)

    # Print the shape of the training fMRI responses and stimulus features: note
    # that the two have the same sample size!
    # print("Training fMRI responses shape:")
    # print(fmri_train.shape)
    # print('(Train samples × Parcels)')
    # print("\nTraining stimulus features shape:")
    # print(features_train.shape)
    # print('(Train samples × Features)')
    
    # # Train the encoding model
    # print("Training the encoding model... ", model_name)
    # model, training_time = train_encoding(features_train, fmri_train)
    # print(f"Training completed in {training_time:.2f} seconds")
    # utils.save_model(model, model_name)

    #model = utils.load_model(f'sub-01_modality-all')
    

if __name__ == "__main__":
    main()