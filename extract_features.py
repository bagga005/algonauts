import utils
import sys
import gzip
import pickle
import os
from glob import glob
from tqdm import tqdm
import h5py
import torch
import numpy as np
from run_experiements import run_trainings
import shutil
#from model_r50_ft import VisionR50FineTuneModel
USE_LIGHTNING = True
# try:
#     from lightning.data import map
# except:
#     print("lightning.data not found, using multiprocessing")
#     USE_LIGHTNING = False
#     from multiprocessing import Pool
from algonaut_funcs import extract_visual_features_r50_ft, extract_visual_features_from_preprocessed_video, load_features, preprocess_features, \
    extract_visual_preprocessed_features, perform_pca, extract_visual_features, get_vision_model, extract_audio_features, get_language_model, \
        define_frames_transform, extract_language_features, perform_pca_incremental
import logging
# import json_log_formatter


# #init logging
# logger = logging.getLogger('my_json')
# formatter = json_log_formatter.JSONFormatter()
# json_handler = logging.FileHandler(filename='run_log-language.json')
# json_handler.setFormatter(formatter)
# logger.addHandler(json_handler)
# logger.setLevel(logging.INFO)

def extract_raw_visual_features_r50_ft(model_name, custom_filter=None):
    pre_features_dir = utils.get_stimulus_pre_features_dir()
    out_data_dir = utils.get_stimulus_features_dir()
# As an exemple, extract visual features for season 1, episode 1 of Friends
    #episode_path = root_data_dir + "algonauts_2025.competitors/stimuli/movies/friends/s1/friends_s01e01a.mkv"
    # Collecting the paths to all the movie stimuli
    file_in_filter = ''
    exclude_list = []#['friends_s03e05b', 'friends_s03e06a']
    if not custom_filter:
        files = glob(f"{pre_features_dir}/pre/visual/*.h5")
    else:
        files = glob(f"{pre_features_dir}/pre/visual/{custom_filter}")
    if file_in_filter:
        files = [f for f in files if file_in_filter in f]
    files.sort()

    stimuli = {f.split("/")[-1].split(".")[0]: f for f in files}
    print(len(stimuli), list(stimuli)[:3], list(stimuli)[-3:])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #setup model once
    model = VisionR50FineTuneModel(8192 * 4, 1000, device)
    params = utils.load_model_pytorch(model_name)
    model.load_state_dict(params)

     # iterate across all the stimuli movie files
    iterator = tqdm(enumerate(stimuli.items()), total=len(list(stimuli)))
    for i, (stim_id, stim_path) in iterator:
        print(f"Extracting visual features for {stim_id}", stim_path)
        fn = os.path.join(out_data_dir, "raw_fit", "visual", f"{stim_id}.h5")
        if os.path.exists(fn) or stim_id in exclude_list: continue; 
        extract_visual_features_r50_ft(stim_path, model, device, fn, stim_id)

def extract_raw_visual_features():
    root_data_dir = utils.get_data_root_dir()
    out_data_dir = utils.get_output_dir()
# As an exemple, extract visual features for season 1, episode 1 of Friends
    #episode_path = root_data_dir + "algonauts_2025.competitors/stimuli/movies/friends/s1/friends_s01e01a.mkv"
    # Collecting the paths to all the movie stimuli
    file_in_filter = 'friends_s03e06a'
    exclude_list = []#['friends_s03e05b', 'friends_s03e06a']
    files = glob(f"{root_data_dir}/algonauts_2025.competitors/stimuli/movies/**/**/*.mkv")
    if file_in_filter:
        files = [f for f in files if file_in_filter in f]
    files.sort()

    stimuli = {f.split("/")[-1].split(".")[0]: f for f in files}
    print(len(stimuli), list(stimuli)[:3], list(stimuli)[-3:])

    # Duration of each movie chunk, aligned with the fMRI TR of 1.49 seconds
    tr = 1.49

    # Saving directories
    save_dir_temp = utils.get_tmp_dir()
    #save_dir_features = out_data_dir +  "stimulus_features/raw/visual/"
    feature_extractor, model_layer, device = get_vision_model()
    transform = define_frames_transform()
    
    # iterate across all the stimuli movie files
    iterator = tqdm(enumerate(stimuli.items()), total=len(list(stimuli)))
    for i, (stim_id, stim_path) in iterator:
        print(f"Extracting visual features for {stim_id}", stim_path)
        fn = os.path.join(out_data_dir, "stimulus_features", "raw", "visual", f"{stim_id}.h5")
        if os.path.exists(fn) or stim_id in exclude_list: continue; 
        # Execute visual feature extraction
        visual_features = extract_visual_features(stim_path, tr, feature_extractor,
        model_layer, transform, device, save_dir_temp, fn, stim_id)

def extract_raw_visual_features_from_preprocessed_video():
    root_data_dir = utils.get_data_root_dir()
    out_data_dir = utils.get_output_dir()
    feature_extractor, model_layer, device = get_vision_model()
    
    group_name = 'visual'
    stim_id = 'friends_s01e24a'
    episode_path = os.path.join(out_data_dir, "stimulus_features", "pre", "visual", f"{stim_id}.h5")
    save_file = os.path.join(out_data_dir, "stimulus_features", "post", "visual", f"{stim_id}.h5")
    extract_visual_features_from_preprocessed_video(episode_path, stim_id, feature_extractor, model_layer, device, save_file, group_name)


def extract_preprocessed_video_content():
    root_data_dir = utils.get_data_root_dir()
    out_data_dir = utils.get_output_dir()
# As an exemple, extract visual features for season 1, episode 1 of Friends
    #episode_path = root_data_dir + "algonauts_2025.competitors/stimuli/movies/friends/s1/friends_s01e01a.mkv"
    # Collecting the paths to all the movie stimuli
    file_in_filter = utils.get_stimuli_prefix()
    exclude_list = []#['friends_s03e05b', 'friends_s03e06a']
    files = glob(f"{root_data_dir}/algonauts_2025.competitors/stimuli/movies/movie10/**/*.mkv")
    print(len(files))
    if file_in_filter:
        # Support comma-separated patterns
        filters = file_in_filter.split(',')
        files = [f for f in files if any(filter_item.strip() in f for filter_item in filters)]
    files.sort()

    stimuli = {f.split("/")[-1].split(".")[0]: f for f in files}
    print(len(stimuli), list(stimuli)[:3], list(stimuli)[-3:])

    # Duration of each movie chunk, aligned with the fMRI TR of 1.49 seconds
    tr = 1.49

    # Saving directories
    save_dir_temp = utils.get_tmp_dir()
    transform = define_frames_transform()
    
    # iterate across all the stimuli movie files
    iterator = tqdm(enumerate(stimuli.items()), total=len(list(stimuli)))
    for i, (stim_id, stim_path) in iterator:
        print(f"Extracting visual features for {stim_id}", stim_path)
        fn = os.path.join(out_data_dir, "stimulus_features", "pre", "visual", f"{stim_id}.h5")
        if os.path.exists(fn) or stim_id in exclude_list: continue; 
        # Execute visual feature extraction
        visual_features = extract_visual_preprocessed_features(stim_path, tr,
        transform, save_dir_temp, fn, stim_id)

def extract_language_for_stimuli(stim_obj, out_data_dir):
    logger.info("Start", extra={'stim_id': stim_obj['stim_id'], 'status': 0})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #save_dir_features = out_data_dir +  "stimulus_features/raw/visual/"
    # Load the model and tokenizer
    model, tokenizer = get_language_model(device)
    extract_language_features(stim_obj['stim_path'], model, tokenizer, stim_obj['num_used_tokens'],
         stim_obj['kept_tokens_last_hidden_state'], device,  stim_obj['fn'], stim_obj['stim_id'])
    logger.info("Finish", extra={'stim_id': stim_obj['stim_id'], 'status': 1})


def extract_raw_language_features():
    root_data_dir = utils.get_data_root_dir()
    out_data_dir = utils.get_output_dir()
    fileFilter = "movie10_life02"
# As an exemple, extract visual features for season 1, episode 1 of Friends
    #episode_path = root_data_dir + "algonauts_2025.competitors/stimuli/movies/friends/s1/friends_s01e01a.mkv"
    # Collecting the paths to all the movie stimuli
    files = glob(f"{root_data_dir}/algonauts_2025.competitors/stimuli/transcripts/**/**/*.tsv")
    if fileFilter:
        files = [f for f in files if fileFilter in f]
    files.sort()

    stimuli = {f.split("/")[-1].split(".")[0]: f for f in files}
    print(len(stimuli), list(stimuli)[:3], list(stimuli)[-3:])

    # Duration of each movie chunk, aligned with the fMRI TR of 1.49 seconds
    tr = 1.49
    # Other parameters
    num_used_tokens = 510
    kept_tokens_last_hidden_state = 10

    
    exclude_list =[]
    stim_list = []
    # iterate across all the stimuli movie files
    iterator = tqdm(enumerate(stimuli.items()), total=len(list(stimuli)))
    for i, (stim_id, stim_path) in iterator:
        print(f"Extracting language features for {stim_id}", stim_path)
        fn = os.path.join(out_data_dir, "stimulus_features", "raw", "language", f"{stim_id}.h5")
        if os.path.exists(fn) or stim_id in exclude_list: continue; 
        stim_obj = dict(stim_path=stim_path, num_used_tokens=num_used_tokens, kept_tokens_last_hidden_state=kept_tokens_last_hidden_state,
                         fn=fn, stim_id=stim_id)
        stim_list.append(stim_obj)
    print('number of stim to process: ' + str(len(stim_list)))
    if USE_LIGHTNING:
        map(
            fn=extract_language_for_stimuli,
            inputs=stim_list,
            num_workers=os.cpu_count(),
            output_dir="thisdic"
        )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, tokenizer = get_language_model(device)
        for stim_obj in stim_list:
            extract_language_features(stim_obj['stim_path'], model, tokenizer, stim_obj['num_used_tokens'],
         stim_obj['kept_tokens_last_hidden_state'], device,  stim_obj['fn'], stim_obj['stim_id'])
 
def extract_audio_for_stimuli(stim_obj, out_data_dir):
    logger.info("Start", extra={'stim_id': stim_obj['stim_id'], 'status': 0})
    extract_audio_features(stim_obj['stim_path'], stim_obj['tr'], stim_obj['sr'], stim_obj['save_dir_temp'], stim_obj['fn'], stim_obj['stim_id'])
    logger.info("Finish", extra={'stim_id': stim_obj['stim_id'], 'status': 1})

def extract_raw_audio_features():
    root_data_dir = utils.get_data_root_dir()
    out_data_dir = utils.get_output_dir()
    tr = 1.49
    # Saving directories
    save_dir_temp = utils.get_tmp_dir()
    #save_dir_features = out_data_dir +  "stimulus_features/raw/visual/"
    sr = 22050
# As an exemple, extract visual features for season 1, episode 1 of Friends
    #episode_path = root_data_dir + "algonauts_2025.competitors/stimuli/movies/friends/s1/friends_s01e01a.mkv"
    # Collecting the paths to all the movie stimuli
    files = glob(f"{root_data_dir}/algonauts_2025.competitors/stimuli/movies/**/**/*.mkv")
    files.sort()

    stimuli = {f.split("/")[-1].split(".")[0]: f for f in files}
    print(len(stimuli), list(stimuli)[:3], list(stimuli)[-3:])

    exclude_list =[]#['friends_s03e05b', 'friends_s03e06a']
    # iterate across all the stimuli movie files
    iterator = tqdm(enumerate(stimuli.items()), total=len(list(stimuli)))
    stim_list = []
    for i, (stim_id, stim_path) in iterator:
        print(f"Extracting audio features for {stim_id}", stim_path)
        # Execute visual feature extraction
        fn = os.path.join(out_data_dir, "stimulus_features", "raw", "audio", f"{stim_id}.h5")
        if os.path.exists(fn) or stim_id in exclude_list: continue; 
        stim_obj = dict(stim_path=stim_path, tr=tr, sr=sr, save_dir_temp=save_dir_temp, fn=fn, stim_id=stim_id)
        stim_list.append(stim_obj)
    map(
        fn=extract_audio_for_stimuli,
        inputs=stim_list,
        num_workers=os.cpu_count(),
        output_dir="thisdic"
    )

def get_shortstim_name(stimuli):
    if 'friends' in stimuli:
        return stimuli[8:]
    elif 'movie10' in stimuli:
        return stimuli[8:]
    else:
        return stimuli
    
def features_combined_npy(infolder, outfile, modality, preProcess=False, zscore=True):
    files = glob(f"{infolder}/*.h5")
    files.sort()
    stimuli = {f.split("/")[-1].split(".")[0]: f for f in files}
    print(len(stimuli), list(stimuli)[:3], list(stimuli)[-3:])
    valdict = {}
    for stim_id, stim_path in stimuli.items():
        features = load_features(stim_path, modality)
        print('shortstim_name: ', get_shortstim_name(stim_id))
        if get_shortstim_name(stim_id) in valdict:
            raise KeyError(f"Key '{get_shortstim_name(stim_id)}' already exists in dictionary")
        if preProcess:
            features = preprocess_features(features, zscore)
        valdict[get_shortstim_name(stim_id)] = features
    data_array = np.array(valdict)
    np.save(outfile, data_array)
    
def reduce_dims_npy(inpath, outfile,  n_components = 250):
    data = np.load(inpath, allow_pickle=True)
    
    boundary = []
    #iterator = tqdm(enumerate(stimuli.items()), total=len(list(stimuli)))
    valdict = {}
    features = []
    data = data.item()
    for stim_id in data.keys():
        #print(f"- {stim_id}")
        fea = np.asarray(data[stim_id])
        #print('fea.shape', fea.shape)
        features.append(fea)
        boundary.append((stim_id, fea.shape[0]))
    features = np.concatenate(features, axis=0)
    print('features.shape', features.shape)

    features_pca = features[:, :n_components]
    print('features_pca.shape', features_pca.shape)
    
    # slice out results
    from_idx =0
    for stim_id, size in boundary:
        #if from_idx < 3000: print(stim_id, size)
        slice = features_pca[from_idx:from_idx+size,:]
        valdict[get_shortstim_name(stim_id)] = slice
        #if from_idx < 3000: print(from_idx, from_idx + size)
        from_idx = from_idx + size
        assert slice.shape[0] == size, "size mismatch while slicing"

    #valdict[get_shortstim_name(stim_id)] = features_pca

    # Convert dictionary to a numpy array (creates a 0-dimensional array containing the dict)
    data_array = np.array(valdict)
    # Save to .npy file
    np.save(outfile, data_array)


def do_pca_npy(inpath, outfile, modality, do_zscore=True, n_components = 250):
    data = np.load(inpath, allow_pickle=True)
    
    boundary = []
    #iterator = tqdm(enumerate(stimuli.items()), total=len(list(stimuli)))
    valdict = {}
    features = []
    data = data.item()
    for stim_id in data.keys():
        #print(f"- {stim_id}")
        fea = np.asarray(data[stim_id])
        #print('fea.shape', fea.shape)
        features.append(fea)
        boundary.append((stim_id, fea.shape[0]))
    features = np.concatenate(features, axis=0)
    print('features.shape', features.shape)
    
    prepr_features = preprocess_features(features, zscore=do_zscore)
    print('prepr_features.shape', prepr_features.shape)


    features_pca = perform_pca(prepr_features, n_components, modality)
    
    # slice out results
    from_idx =0
    for stim_id, size in boundary:
        #if from_idx < 3000: print(stim_id, size)
        slice = features_pca[from_idx:from_idx+size,:]
        valdict[get_shortstim_name(stim_id)] = slice
        #if from_idx < 3000: print(from_idx, from_idx + size)
        from_idx = from_idx + size
        assert slice.shape[0] == size, "size mismatch while slicing"

    #valdict[get_shortstim_name(stim_id)] = features_pca

    # Convert dictionary to a numpy array (creates a 0-dimensional array containing the dict)
    data_array = np.array(valdict)
    # Save to .npy file
    np.save(outfile, data_array)
    
    
    

def do_pca(inpath, outfile,modality, do_zscore=True,skip_pca_just_comgine=False, n_components = 250):
    root_data_dir = utils.get_data_root_dir()
    out_data_dir = utils.get_output_dir()
    
    files = glob(f"{inpath}/*.h5")

    files.sort()

    stimuli = {f.split("/")[-1].split(".")[0]: f for f in files}

    #fileFilter = "movie10_life02"
    boundary = []
    iterator = tqdm(enumerate(stimuli.items()), total=len(list(stimuli)))
    valdict = {}
    features = []
    for i, (stim_id, stim_path) in iterator:
        with h5py.File(stim_path, 'r') as f1:
            #print("Root level keys:", list(f1.keys()))
            #data = f1[stim_id]['language_last_hidden_state'][:]
            #print(data.shape)
            fea = load_features(stim_path, modality)
            #print('fea.shape', fea.shape)
            features.append(fea)
            boundary.append((stim_id, fea.shape[0]))
            #print(get_shortstim_name(stim_id))
            #print('extracted features.shape', features.shape)
            # Preprocess the stimulus features
    features = np.concatenate(features, axis=0)
    print('features.shape', features.shape)
    
    if not skip_pca_just_comgine:
        prepr_features = preprocess_features(features, zscore=do_zscore)
        print('prepr_features.shape', prepr_features.shape)

        if (prepr_features.shape[1] > 17920):
            # Perform Incremental PCA
            features_pca = perform_pca_incremental(prepr_features, n_components, modality)
        else:
            # Perform Classic PCA
            features_pca = perform_pca(prepr_features, n_components, modality)
        #print('pca features.shape', features_pca.shape)
    else:
        features_pca = features

    # slice out results
    from_idx =0
    for stim_id, size in boundary:
        #if from_idx < 3000: print(stim_id, size)
        slice = features_pca[from_idx:from_idx+size,:]
        valdict[get_shortstim_name(stim_id)] = slice
        #if from_idx < 3000: print(from_idx, from_idx + size)
        from_idx = from_idx + size
        assert slice.shape[0] == size, "size mismatch while slicing"

    #valdict[get_shortstim_name(stim_id)] = features_pca

    # Convert dictionary to a numpy array (creates a 0-dimensional array containing the dict)
    data_array = np.array(valdict)
    # Save to .npy file
    np.save(outfile, data_array)
    # Collecting the paths to all the movie stimuli
    #files = glob(f"{root_data_dir}algonauts_2025.competitors/stimulus_features/raw/language/*.h5")
def compute_tr_upper(dir_path, stim_id, layer_name):
    files = glob(f"{dir_path}/{stim_id}_tr_*_{layer_name}.pt.gz")
    return len(files)

def segment_to_extract(loaded_tensor, combine_strategy, i=0, j=0,indexes=[]):
    if combine_strategy == COMBINE_STRATEGY_LAST:
        ten = loaded_tensor[-1,:]
    elif combine_strategy == COMBINE_STRATEGY_LAST3:
        if loaded_tensor.shape[0] < 3:
            gap = 3 - loaded_tensor.shape[0]
            last_dim = loaded_tensor[0,:]
            gap_tensor = last_dim.repeat(gap, 1)
            ten_new = torch.cat((gap_tensor, loaded_tensor), dim=0)
        else:
            ten_new = loaded_tensor
        ten = ten_new[-3:, :].flatten()
    elif combine_strategy == COMBINE_STRATEGY_LAST7:
        #if input is short, you may not have 7 tokens. Copy the last token to make up for the missing dimensions
        if loaded_tensor.shape[0] < 7:
            gap = 7 - loaded_tensor.shape[0]
            last_dim = loaded_tensor[0,:]
            gap_tensor = last_dim.repeat(gap, 1)
            ten_new = torch.cat((gap_tensor, loaded_tensor), dim=0)
        else:
            ten_new = loaded_tensor
        ten = ten_new[-7:, :].flatten()
    elif combine_strategy == COMBINE_STRATEGY_LAST10:
        ten = torch.cat((loaded_tensor[-1,:], loaded_tensor[-2,:], loaded_tensor[-3,:], loaded_tensor[-4,:], loaded_tensor[-5,:], loaded_tensor[-6,:], loaded_tensor[-7,:], loaded_tensor[-8,:], loaded_tensor[-9,:], loaded_tensor[-10,:]), dim=0)
        # print('ten.shape', ten.shape)
    elif combine_strategy == COMBINE_STRATEGY_LAST7_AVG:
        ten = torch.mean(loaded_tensor[-7:,:], dim=0)
        # print('ten.shape', ten.shape)
    elif combine_strategy == COMBINE_STRATEGY_FIRST:
        ten = loaded_tensor[0,:]
    elif combine_strategy == COMBINE_STRATEGY_I:
        #regular code
        ten = loaded_tensor[i,:]
    elif combine_strategy == COMBINE_STRATEGY_VISION_V2:
        ##special handling of vision for season 1
        id1, id2, id3 = 0, 0, 8
        if i == 1:
            id1, id2, id3 = 1, 8, 16

        ten = loaded_tensor[id2:id3,:].flatten()
        print('ten.shape', ten.shape)
    elif combine_strategy == COMBINE_STRATEGY_I_J:
        ten = loaded_tensor[i:j+1,:]
        ten = ten.flatten()
    elif combine_strategy == COMBINE_STRATEGY_INDEXS:
        ten = loaded_tensor[indexes,:].flatten()
    elif combine_strategy == COMBINE_STRATEGY_I_J_AVG:
        ten = torch.mean(loaded_tensor[i:j+1,:], dim=0)
    elif combine_strategy == COMBINE_STRATEGY_INDEXS_AVG:
        ten = torch.mean(loaded_tensor[indexes,:], dim=0)
    else:
        raise ValueError(f"Invalid strategy: {combine_strategy}")
    
    return ten

def segment_to_extract_v2(loaded_tensor, combine_strategy, i=0, j=0,indexes=[]):
    if combine_strategy == COMBINE_STRATEGY_LAST:
        ten = loaded_tensor[:,-1,:]
    elif combine_strategy == COMBINE_STRATEGY_LAST3:
        selected = loaded_tensor[:,-3:,:]
        ten = selected.reshape(selected.shape[0], -1)

    elif combine_strategy == COMBINE_STRATEGY_LAST7:
        selected = loaded_tensor[:,-7:,:]
        ten = selected.reshape(selected.shape[0], -1)
    else:
        raise ValueError(f"Invalid strategy: {combine_strategy}")
    
    return ten

def get_stim_id_list(dir_path, filter_in_name=None, add_layer_to_path=True):
    if add_layer_to_path:
        dir_path = os.path.join(dir_path, 'metadata')
    files = glob(f"{dir_path}/*_metadata.json")
    f_list = [f.split("/")[-1].split("_")[0] + "_" + f.split("/")[-1].split("_")[1] for f in files]
    f_list = list(set(f_list))
    f_list.sort()
    #print(filter_in_name)

    if filter_in_name is not None:
        # Keep files that contain any of the strings in filter_in_name
        f_list = [f for f in f_list if any(filter_str in f for filter_str in filter_in_name)]

    return f_list

#LLM Only
STRATEGY_LANG_NORM_1 = 0
STRATEGY_LANG_NORM_3 = 1
STRATEGY_LANG_4_12_NORM = 2
STRATEGY_LANG_NORM_7 = 7
STRATEGY_LANG_NORM_10 = 8
STRATEGY_LANG_NORM_7_AVG = 9
STRATEGY_V2_LANG_NORM_1 = 100
STRATEGY_V2_LANG_NORM_3 = 101
STRATEGY_V2_LANG_NORM_5 = 102
STRATEGY_V2_LANG_NORM_7 = 103
STRATEGY_V2_LANG_NORM_AVG_PRE = 104
STRATEGY_V2_LANG_NORM_AVG_POST = 105
STRATEGY_V2_LANG_NORM_FIRST = 106
STRATEGY_V2_IMGPLUS1 = 107  
STRATEGY_V2_POST_LAST = 108
STRATEGY_V2_POST_L4_LAST = 109
STRATEGY_V2_LANG_NORM_I1 = 110
STRATEGY_V2_LANG_NORM_I2 = 111
STRATEGY_V2_PRE_LAST = 112
STRATEGY_V2_PRE_LAST5 = 113
STRATEGY_V2_PRE_LAST5_AVG = 114
STRATEGY_V2_IMG8A = 115
STRATEGY_V2_LN7_IMG8A = 116
STRATEGY_V3_MP_LANG_NORM_7 = 500
STRATEGY_V3_FPMP_LANG_NORM_1 = 499
STRATEGY_V3_FP_LANG_NORM_7 = 501
STRATEGY_V3_FP_LANG_NORM_1 = 502
STRATEGY_V3_MP_LANG_NORM_1 = 503
STRATEGY_V3_MP_PRE_7 = 504
STRATEGY_V3_MP_POST_F3 = 505
STRATEGY_V3_MP_POST_L4 = 506
STRATEGY_V3_MP_POST_F3_AVG = 507
STRATEGY_V3_MP_POST_L4_AVG = 508
STRATEGY_V3_IMG8A = 513
STRATEGY_V3_IMG_PLUS1 = 514
STRATEGY_V4_IMG8A = 550
STRATEGY_V4_IMG_PLUS1 = 551
STRATEGY_V4_POST_L3_AVG = 552
STRATEGY_V4_POST_L7_AVG = 553
STRATEGY_V4_POST_L10_AVG = 554
STRATEGY_V4_POST_LALL_AVG = 555
STRATEGY_V4_POST_4Layer_L10_AVG = 556
STRATEGY_V4_POST_L12_L10_AVG = 557
STRATEGY_V4_POST_5Layer_L10_AVG = 558
STRATEGY_V4_POST_L4_L10_AVG = 559
STRATEGY_V4_POST_5Layer_L10_AVG_IMG8A = 560

#Vision 
STRATEGY_VISION_NORM = 10
STRATEGY_VISION_23= 11
STRATEGY_VISION_2_5_10_17_NORM = 12
STRATEGY_V2_VISION_NORM_CLS = 200
STRATEGY_V2_VISION_NORM_AVG = 201
STRATEGY_V4_L12_CLS = 301
STRATEGY_V4_L4_CLS = 302
STRATEGY_V4_L2_CLS = 303
STRATEGY_V4_LNORM_CLS = 304
STRATEGY_V4_LNORM_12_4_CLS = 305
STRATEGY_V4_L22_CLS = 306
STRATEGY_V4_L23_CLS = 307
#STRATEGY_V2_VISION_

#LLM + Vision
STRATEGY_LN_1_VN = 20
STRATEGY_LN_3_VN = 21
STRATEGY_LN7_4_12_NORM_VN_NORM = 22
STRATEGY_V2_LN7_VCLS = 300


COMBINE_STRATEGY_LAST = 'last'
COMBINE_STRATEGY_LAST3 = 'last3'
COMBINE_STRATEGY_LAST4 = 'last4'
COMBINE_STRATEGY_LAST5 = 'last5'
COMBINE_STRATEGY_LAST6 = 'last6'
COMBINE_STRATEGY_LAST7 = 'last7'
COMBINE_STRATEGY_LAST8 = 'last8'
COMBINE_STRATEGY_LAST9 = 'last9'
COMBINE_STRATEGY_LAST10 = 'last10'
COMBINE_STRATEGY_LAST7_AVG = 'last7_avg'
COMBINE_STRATEGY_FIRST = 'first'
COMBINE_STRATEGY_I ='ith'
COMBINE_STRATEGY_I_J = 'ith_jth'
COMBINE_STRATEGY_I_J_AVG = 'ith_jth_avg'
COMBINE_STRATEGY_INDEXS = 'indexs'
COMBINE_STRATEGY_INDEXS_AVG = 'indexs_avg'
COMBINE_STRATEGY_VISION_V2 = 'vision_v2'

def save_combined_vlm_features(dir_input_path, dir_output_path, strategy, modality, filter_in_name=None, overwrite=False, add_layer_to_path=True):
    stim_id_list = get_stim_id_list(dir_input_path, filter_in_name, add_layer_to_path)
    num_skipped = 0
    e_format = utils.get_embeddings_format()
    if e_format != '2' and e_format != '1':
        raise ValueError(f"Invalid embeddings format: {e_format}")
    with tqdm(total=len(stim_id_list), desc="Saving combined VLM features for {}".format(strategy)) as pbar:
        for stim_id in stim_id_list:
            save_file = os.path.join(dir_output_path, f"{stim_id}.h5")
            if not overwrite and os.path.exists(save_file):
                pbar.update(1)
                num_skipped += 1
                continue
            if strategy == STRATEGY_V3_MP_LANG_NORM_7:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_I_J,i=7,j=13)
            elif strategy == STRATEGY_V3_FP_LANG_NORM_7:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_I_J,i=0,j=6)
            elif strategy == STRATEGY_V3_FP_LANG_NORM_1:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_I,i=6)
            elif strategy == STRATEGY_V3_MP_LANG_NORM_1:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_I,i=13)                
            elif strategy == STRATEGY_V4_POST_L3_AVG:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_I,i=7)
            elif strategy == STRATEGY_V4_POST_L7_AVG:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_I,i=8)
            elif strategy == STRATEGY_V4_POST_L10_AVG:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_I,i=9)
            elif strategy == STRATEGY_V4_POST_L12_L10_AVG:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_layers_12", COMBINE_STRATEGY_I,i=10)
            elif strategy == STRATEGY_V4_POST_L4_L10_AVG:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_layers_4", COMBINE_STRATEGY_I,i=10)
            elif strategy == STRATEGY_V4_POST_LALL_AVG:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_I,i=10)
            elif strategy == STRATEGY_V4_IMG_PLUS1:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_I,i=19)

            
            elif strategy == STRATEGY_V3_MP_PRE_7:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_I_J,i=0,j=6)
            elif strategy == STRATEGY_V3_MP_POST_F3:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_I_J,i=7,j=9)
            elif strategy == STRATEGY_V3_MP_POST_L4:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_I_J,i=10,j=13)
            elif strategy == STRATEGY_V3_MP_POST_F3_AVG:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_I_J_AVG,i=7,j=9)
            elif strategy == STRATEGY_V3_MP_POST_L4_AVG:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_I_J_AVG,i=10,j=13)
            elif strategy == STRATEGY_V3_IMG_PLUS1:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_I,i=22)    
            elif strategy == STRATEGY_LANG_NORM_1:
                if e_format == '1':
                    ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_LAST, add_layer_to_path)
                elif e_format == '2':
                    ten1 = combine_vlm_features(dir_input_path, stim_id, -1, COMBINE_STRATEGY_LAST)
                    print('ten1.shape', ten1.shape)
            elif strategy == STRATEGY_LANG_NORM_3:
                if e_format == '1':
                    ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_LAST3, add_layer_to_path)
                elif e_format == '2':
                    ten1 = combine_vlm_features(dir_input_path, stim_id, -1, COMBINE_STRATEGY_LAST3)
                    print('ten1.shape', ten1.shape)
            elif strategy == STRATEGY_LANG_NORM_7:
                if e_format == '1':
                    ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_LAST7, add_layer_to_path)
                elif e_format == '2':
                    ten1 = combine_vlm_features(dir_input_path, stim_id, -1, COMBINE_STRATEGY_LAST7)
                #print('ten1.shape', ten1.shape)
            elif strategy == STRATEGY_LANG_NORM_10:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_LAST10, add_layer_to_path)
            elif strategy == STRATEGY_LANG_NORM_7_AVG:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_LAST7_AVG)
                # print('ten1.shape', ten1.shape)
                # ten1 = torch.mean(ten1, dim=1)
                # print('ten1.shape mean', ten1.shape)
            elif strategy == STRATEGY_VISION_NORM:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "vision_model", COMBINE_STRATEGY_FIRST, add_layer_to_path)
            elif strategy == STRATEGY_VISION_23:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "vision_model_encoder_layers_23", COMBINE_STRATEGY_FIRST, add_layer_to_path)
            elif strategy == STRATEGY_VISION_2_5_10_17_NORM:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "vision_model_encoder_layers_2", COMBINE_STRATEGY_FIRST, add_layer_to_path)
                ten2 = combine_vlm_features(dir_input_path, stim_id, "vision_model_encoder_layers_5", COMBINE_STRATEGY_FIRST, add_layer_to_path)
                ten3 = combine_vlm_features(dir_input_path, stim_id, "vision_model_encoder_layers_10", COMBINE_STRATEGY_FIRST, add_layer_to_path)
                ten4 = combine_vlm_features(dir_input_path, stim_id, "vision_model_encoder_layers_17", COMBINE_STRATEGY_FIRST, add_layer_to_path)
                ten5 = combine_vlm_features(dir_input_path, stim_id, "vision_model", COMBINE_STRATEGY_FIRST, add_layer_to_path)
                ten1 = torch.cat((ten1, ten2, ten3, ten4, ten5), dim=1)
                #print('ten1.shape', ten1.shape)
            elif strategy == STRATEGY_LN_1_VN:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_LAST)
                ten2 = combine_vlm_features(dir_input_path, stim_id, "vision_model", COMBINE_STRATEGY_FIRST)
                ten1 = torch.cat((ten1, ten2), dim=1)
            elif strategy == STRATEGY_LN_3_VN:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_LAST3)
                ten2 = combine_vlm_features(dir_input_path, stim_id, "vision_model", COMBINE_STRATEGY_FIRST)
                ten1 = torch.cat((ten1, ten2), dim=1)
            elif strategy == STRATEGY_V4_POST_4Layer_L10_AVG:
                ten2 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_I,i=9)
                ten3 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_layers_23", COMBINE_STRATEGY_I,i=9)
                ten4 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_layers_22", COMBINE_STRATEGY_I,i=9)
                ten5 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_layers_21", COMBINE_STRATEGY_I,i=9)
                ten1 = torch.cat((ten2, ten3, ten4, ten5), dim=1)
            elif strategy == STRATEGY_V4_POST_5Layer_L10_AVG:
                ten2 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_I,i=9)
                ten3 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_layers_23", COMBINE_STRATEGY_I,i=9)
                ten4 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_layers_22", COMBINE_STRATEGY_I,i=9)
                ten5 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_layers_21", COMBINE_STRATEGY_I,i=9)
                ten6 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_layers_12", COMBINE_STRATEGY_I,i=9)
                ten1 = torch.cat((ten2, ten3, ten4, ten5, ten6), dim=1)
            elif strategy == STRATEGY_V4_POST_5Layer_L10_AVG_IMG8A:
                ten2 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_I,i=9)
                ten3 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_layers_23", COMBINE_STRATEGY_I,i=9)
                ten4 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_layers_22", COMBINE_STRATEGY_I,i=9)
                ten5 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_layers_21", COMBINE_STRATEGY_I,i=9)
                ten6 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_layers_12", COMBINE_STRATEGY_I,i=9)
                ten7 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_I_J, i=11, j=18)
                ten1 = torch.cat((ten2, ten3, ten4, ten5, ten6, ten7), dim=1)
            elif strategy == STRATEGY_LN7_4_12_NORM_VN_NORM:
                ten2 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_layers_4", COMBINE_STRATEGY_LAST7)
                ten3 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_layers_12", COMBINE_STRATEGY_LAST7)
                ten4 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_LAST7)
                ten5 = combine_vlm_features(dir_input_path, stim_id, "vision_model", COMBINE_STRATEGY_FIRST)
                ten1 = torch.cat((ten2, ten3, ten4, ten5), dim=1)
                #print('combined_tensor.shape', ten1.shape)
            elif strategy == STRATEGY_LANG_4_12_NORM:
                ten0 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_layers_4", COMBINE_STRATEGY_LAST)
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_layers_12", COMBINE_STRATEGY_LAST)
                ten2 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_LAST)
                ten1 = torch.cat((ten0, ten1, ten2), dim=1)
            elif strategy == STRATEGY_V2_LANG_NORM_1:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_I, i=6)
            elif strategy == STRATEGY_V2_LANG_NORM_3:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_I_J,i=0,j=2)
            elif strategy == STRATEGY_V2_LANG_NORM_5:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_I_J,i=0,j=4)
            elif strategy == STRATEGY_V2_LANG_NORM_7:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_I_J, i=0,j=6)
                #print('ten1.shape', ten1.shape)
            elif strategy == STRATEGY_V2_LANG_NORM_I1:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_I, i=1)
            elif strategy == STRATEGY_V2_LANG_NORM_I2:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_I, i=2)
            elif strategy == STRATEGY_V2_PRE_LAST:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_I, i=15)
            elif strategy == STRATEGY_V2_PRE_LAST5:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_I_J, i=19,j=23)
            elif strategy == STRATEGY_V2_PRE_LAST5_AVG:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_I_J_AVG, i=19,j=23)
            elif strategy == STRATEGY_V2_LANG_NORM_AVG_PRE:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_I, i=8)
            elif strategy == STRATEGY_V2_LANG_NORM_AVG_POST:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_I, i=16)
            elif strategy == STRATEGY_V2_POST_LAST:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_I, i=23)
            elif strategy == STRATEGY_V2_POST_L4_LAST:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_I, i=23)
                ten2 = combine_vlm_features(dir_input_path, stim_id, "language_model.model.layers.23", COMBINE_STRATEGY_I, i=23)
                ten3 = combine_vlm_features(dir_input_path, stim_id, "language_model.model.layers.22", COMBINE_STRATEGY_I, i=23)
                ten4 = combine_vlm_features(dir_input_path, stim_id, "language_model.model.layers.21", COMBINE_STRATEGY_I, i=23)
                ten5 = combine_vlm_features(dir_input_path, stim_id, "language_model.model.layers.20", COMBINE_STRATEGY_I, i=23)
                ten6 = torch.cat((ten2, ten3, ten4, ten5), dim=0)
                ten1 = torch.cat((ten1, ten6), dim=0)
            elif strategy == STRATEGY_V2_LANG_NORM_FIRST:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_I, i=7)
            elif strategy == STRATEGY_V2_IMGPLUS1:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_I, i=32)
            elif strategy == STRATEGY_V2_IMG8A:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_I_J, i=24, j=31)
            elif strategy == STRATEGY_V4_IMG8A:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_I_J, i=11, j=18)
            elif strategy == STRATEGY_V2_VISION_NORM_CLS:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "vision_model", COMBINE_STRATEGY_VISION_V2, i=0)
            elif strategy == STRATEGY_V4_LNORM_CLS:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "vision_model", COMBINE_STRATEGY_I_J, i=0, j=7)
            elif strategy == STRATEGY_V4_L12_CLS:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "vision_model_encoder_layers_12", COMBINE_STRATEGY_I_J, i=0, j=7)
            elif strategy == STRATEGY_V4_L4_CLS:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "vision_model_encoder_layers_4", COMBINE_STRATEGY_I_J, i=0, j=7)
            elif strategy == STRATEGY_V4_L22_CLS:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "vision_model_encoder_layers_22", COMBINE_STRATEGY_I_J, i=0, j=7)
            elif strategy == STRATEGY_V4_L23_CLS:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "vision_model_encoder_layers_23", COMBINE_STRATEGY_I_J, i=0, j=7)
            elif strategy == STRATEGY_V4_LNORM_12_4_CLS:
                ten2 = combine_vlm_features(dir_input_path, stim_id, "vision_model", COMBINE_STRATEGY_I_J, i=0, j=7)
                ten3 = combine_vlm_features(dir_input_path, stim_id, "vision_model_encoder_layers_12", COMBINE_STRATEGY_I_J, i=0, j=7)
                ten4 = combine_vlm_features(dir_input_path, stim_id, "vision_model_encoder_layers_4", COMBINE_STRATEGY_I_J, i=0, j=7)
                ten1 = torch.cat((ten2, ten3, ten4), dim=1)
            elif strategy == STRATEGY_V4_L2_CLS:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "vision_model_encoder_layers_2", COMBINE_STRATEGY_I_J, i=0, j=7)
            elif strategy == STRATEGY_V2_VISION_NORM_AVG:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "vision_model", COMBINE_STRATEGY_VISION_V2, i=1)
            elif strategy == STRATEGY_V2_LN7_IMG8A:
                ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_I_J, i=0,j=6)
                ten2 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_I_J, i=24, j=31)
                ten1 = torch.cat((ten1, ten2), dim=1)
            else:
                raise ValueError(f"Invalid strategy: {strategy}")

            if e_format == '2':
                assert ten1.dtype == np.float16, f"ten1.dtype {ten1.dtype} != torch.float16"
                visual_features = ten1.astype(np.float32)
            else:
                assert ten1.dtype == torch.float32, f"ten1.dtype {ten1.dtype} != torch.float32"
                visual_features = ten1.cpu().numpy()
            
            with h5py.File(save_file, 'w') as f:
                group = f.create_group(stim_id)
                group.create_dataset(modality, data=visual_features, dtype=np.float32)   
            pbar.update(1)
        if num_skipped > 0:
            print(f"****Skipped {num_skipped} files")
        
def combine_vlm_features(dir_path, stim_id, layer_name, strategy, add_layer_to_path=True, i=0, j=0, indexes=[]):
    e_format = utils.get_embeddings_format()
    if e_format == '1':
        tensor_list = []
        if add_layer_to_path:
            dir_path = os.path.join(dir_path, layer_name)
        tr_upper = compute_tr_upper(dir_path, stim_id, layer_name)
        #print('tr_upper', stim_id, tr_upper)
        
        #print('dir_path', dir_path)
        for tr_i in range(tr_upper):
            file_path = os.path.join(dir_path, f"{stim_id}_tr_{tr_i}_{layer_name}.pt.gz")
            with gzip.open(file_path, 'rb') as f:
                loaded_tensor = pickle.load(f)
            #assert loaded_tensor.dtype == torch.bfloat16, f"loaded_tensor.dtype {loaded_tensor.dtype} != torch.bfloat16"
            if loaded_tensor.dtype == torch.bfloat16:
                loaded_tensor = loaded_tensor.to(torch.float32)
            assert loaded_tensor.dtype == torch.float32, f"loaded_tensor.dtype {loaded_tensor.dtype} != torch.float32"
            extracted_tensor = segment_to_extract(loaded_tensor, strategy, i, j, indexes)
            #print('extracted_tensor.dtype', file_path, extracted_tensor.dtype)
            tensor_list.append(extracted_tensor)
            #print('tensor_list.shape', len(tensor_list))
        combined_tensor = torch.stack(tensor_list)
    elif e_format == '2':
        file_path = os.path.join(dir_path, f"{stim_id}.npy")
        if os.path.exists(file_path):
            activations = np.load(file_path)
            # print('activations.shape', activations.shape)
            combined_tensor = segment_to_extract_v2(activations[-1], strategy, i, j, indexes)
            # print('combined_tensor.shape', combined_tensor.shape)
        else:
            raise ValueError(f"File not found: {file_path}")
    #print(f"combine_features: {strategy}, combined_tensor.shape", combined_tensor.shape, "stim_id", stim_id)
    return combined_tensor


def perform_pca_evaluate_embeddings(strategy, strategy_name, pca_dim, modality, skip_evaluation, dir_output_path, overwrite=False, force_evaluation=False):
    pca_file_path =os.path.join(dir_output_path, f"features_train-{pca_dim}.npy")
    if not os.path.exists(pca_file_path) or overwrite:
        do_pca(dir_output_path, pca_file_path, modality, do_zscore=True, skip_pca_just_comgine=False, n_components=pca_dim)
    else:
        print(f"**Skipping pca for {strategy_name} {pca_dim} because file already exists")
    if not skip_evaluation:
        #get results file path to see if it exists
        eval_dir = os.path.join(dir_output_path, 'evals')
        results_file_path = utils.get_subject_network_accuracy_file_for_experiement(strategy_name+'-'+str(pca_dim), eval_dir)
        if not os.path.exists(results_file_path) or overwrite or force_evaluation:
        #move generated file to pca directory
            stim_file_path = os.path.join(utils.get_stimulus_features_dir(), 'pca', 'friends_movie10', 'visual', 'features_train.npy')
            shutil.copy(pca_file_path, stim_file_path)
            #run evaluation
            run_trainings(experiment_name=strategy_name+'-'+str(pca_dim), results_output_directory=eval_dir)
        else:
            print(f"**Skipping validation for {strategy_name} {pca_dim} because results file already exists")

def exec_emb_and_pca(dir_input_path, dir_output_path, strategy_name, strategy, modality, filter_in_name=None, pca_only=False, pca_skip=False, overwrite=False, add_layer_to_path=True,  skip_evaluation=False, overwrite_pca=False, force_evaluation=False, pca_dims=[250,500,1000]):
    os.makedirs(dir_output_path, exist_ok=True)	
    if not pca_only:
        print(f"\n**Starting save_combined_vlm_features for {strategy_name}")
        save_combined_vlm_features(dir_input_path, dir_output_path, strategy, modality, filter_in_name, overwrite, add_layer_to_path)
    if not pca_skip:
        pca_file_path =os.path.join(dir_output_path, f"features_train.npy")
        if not os.path.exists(pca_file_path) or overwrite_pca:
            print(f"**Starting pca for {strategy_name}")
            do_pca(dir_output_path, pca_file_path, modality, do_zscore=False, skip_pca_just_comgine=True)
        for pca_dim in pca_dims:
            perform_pca_evaluate_embeddings(strategy, strategy_name, pca_dim, modality, skip_evaluation, dir_output_path, overwrite_pca, force_evaluation)


def get_embeddings_and_evaluate_for_strategy(strategy_folder_name, strategy_id, dir_input_path, dir_output_path, **kwargs):
    
    dir_output_path_strategy = os.path.join(dir_output_path, strategy_folder_name)
    exec_emb_and_pca(dir_input_path, dir_output_path_strategy, strategy_folder_name, strategy_id, **kwargs)
    
if __name__ == "__main__":
    out_dir = utils.get_output_dir()
    embeddings_dir = utils.get_embeddings_dir()
    dir_input_path = os.path.join(out_dir, embeddings_dir)
    DIR_INPUT_PATH_OLD = os.path.join(out_dir, "embeddings")
    embeddings_combined_dir = utils.get_embeddings_combined_dir()
    dir_output_path = os.path.join(out_dir, embeddings_combined_dir)

    filter_in_name = ["s01", "s02", "s03", "s04", "s05", "s06"]
    # filter_in_name = [ "s02","s03", "s04",  "s06"]
    #filter_in_name = ["s03", "s04", "s05", "s06"]
    modality = "visual"
    
    strategy ="STRATEGY_V4_POST_L12_L10_AVG"
    
    if len(sys.argv) > 1:
        # Check if first argument is -combine
        if sys.argv[1] == "-combine":
            if len(sys.argv) != 4:
                print("Usage for combine mode: python extract_features.py -combine pca network")
                sys.exit(1)
            pca_dim = sys.argv[2]
            network = sys.argv[3]
            print(f"Calling utils.consolidate_results({pca_dim}, {network})")
            utils.consolidate_results(pca_dim, network)
        else:
            # Original logic for processing strategies
            for arg in sys.argv[1:]:
                strategy = arg
                strategy_id = globals()[strategy]
                kwargs = dict(modality=modality, filter_in_name=filter_in_name, \
                    #overwrite_pca=True, \
                    #overwrite=True, \
                    #pca_skip=True \
                    # force_evaluation=True \
                    pca_dims=[250, 500]
                    )
                get_embeddings_and_evaluate_for_strategy(strategy, strategy_id, \
            dir_input_path, dir_output_path, **kwargs)  
    #inpath = "/home/bagga005/algo/comp_data/stimulus_features/raw/language/friends_s01e01a_features_language.h5"
    # stim_folder = utils.get_stimulus_features_dir()
    # inpath = os.path.join(stim_folder, 'pca', 'friends_movie10', 'audio', 'features_train.npy')
    # #inpath = "/home/bagga005/algo/comp_data/stimulus_features/pca/friends_movie10/language/features_train.npy"
    # outfile = os.path.join(stim_folder, 'pca', 'friends_movie10', 'audio', 'features_train-250.npy')
    # do_pca_npy(inpath, outfile, modality, do_zscore=True,skip_pca_just_comgine=False, n_components = 250)
    # outfile = os.path.join(stim_folder, 'pca', 'friends_movie10', 'audio', 'features_train-500.npy')
    # do_pca_npy(inpath, outfile, modality, do_zscore=True,skip_pca_just_comgine=False, n_components = 500)
    # outfile = os.path.join(stim_folder, 'pca', 'friends_movie10', 'audio', 'features_train-1000.npy')
    # do_pca_npy(inpath, outfile, modality, do_zscore=True,skip_pca_just_comgine=False, n_components = 1000)
    
    
    
    # combine_vlm_features(dir_input_path, "friends_s04e20b", -1, COMBINE_STRATEGY_LAST, overwrite=True)
    # save_combined_vlm_features(dir_input_path, dir_output_path, strategy, modality, filter_in_name=None, overwrite=False, add_layer_to_path=True):

    # STRATEGY_LANG_NORM_1 v1 - OOOOLLLLDDD
    # dir_output_path = os.path.join(dir_output_path, "STRATEGY_LANG_NORM_1_v1")
    # exec_emb_and_pca(DIR_INPUT_PATH_OLD, dir_output_path, STRATEGY_LANG_NORM_1, modality, filter_in_name=filter_in_name, overwrite=True, add_layer_to_path=False)

    # # #STRATEGY_V2_LANG_NORM_3
    # dir_output_path_me = os.path.join(dir_output_path, "STRATEGY_V2_IMGPLUS1")
    # exec_emb_and_pca(dir_input_path, dir_output_path_me, STRATEGY_V2_IMGPLUS1, modality, filter_in_name=filter_in_name)

    # #STRATEGY_V2_LANG_NORM_1
    # dir_output_path_me = os.path.join(dir_output_path, "STRATEGY_V3_MP_LANG_NORM_7")
    # exec_emb_and_pca(dir_input_path, dir_output_path_me, STRATEGY_V3_MP_LANG_NORM_7, modality, filter_in_name=filter_in_name, overwrite=True)


    # #STRATEGY_V2_LANG_NORM_5
    # dir_output_path_me = os.path.join(dir_output_path, "STRATEGY_V3_FP_LANG_NORM_7")
    # exec_emb_and_pca(dir_input_path, dir_output_path_me, STRATEGY_V3_FP_LANG_NORM_7, modality, filter_in_name=filter_in_name, overwrite=True)

    # # #STRATEGY_V2_LANG_NORM_5
    # dir_output_path_me = os.path.join(dir_output_path, "STRATEGY_V4_IMG8A")
    # exec_emb_and_pca(dir_input_path, dir_output_path_me, STRATEGY_V4_IMG8A, modality, filter_in_name=filter_in_name)

    
    # # # #STRATEGY_V2_LANG_NORM_5
    # dir_output_path_me = os.path.join(dir_output_path, "STRATEGY_V3_MP_LANG_NORM_1")
    # exec_emb_and_pca(dir_input_path, dir_output_path_me, STRATEGY_V3_MP_LANG_NORM_1, modality, filter_in_name=filter_in_name)
    
    # dir_output_path_me = os.path.join(dir_output_path, "STRATEGY_V3_IMG_PLUS1")
    # exec_emb_and_pca(dir_input_path, dir_output_path_me, STRATEGY_V3_IMG_PLUS1, modality, filter_in_name=filter_in_name)
    
    # dir_output_path_me = os.path.join(dir_output_path, "STRATEGY_V3_MP_PRE_7")
    # exec_emb_and_pca(dir_input_path, dir_output_path_me, STRATEGY_V3_MP_PRE_7, modality, filter_in_name=filter_in_name)
    
    # dir_output_path_me = os.path.join(dir_output_path, "STRATEGY_V3_MP_POST_F3")
    # exec_emb_and_pca(dir_input_path, dir_output_path_me, STRATEGY_V3_MP_POST_F3, modality, filter_in_name=filter_in_name)
    
    # dir_output_path_me = os.path.join(dir_output_path, "STRATEGY_V3_MP_POST_L4")
    # exec_emb_and_pca(dir_input_path, dir_output_path_me, STRATEGY_V3_MP_POST_L4, modality, filter_in_name=filter_in_name)
    
    # dir_output_path_me = os.path.join(dir_output_path, "STRATEGY_V4_POST_L3_AVG")
    # exec_emb_and_pca(dir_input_path, dir_output_path_me, STRATEGY_V4_POST_L3_AVG, modality, filter_in_name=filter_in_name)
    
    # dir_output_path_me = os.path.join(dir_output_path, "STRATEGY_V4_POST_L7_AVG")
    # exec_emb_and_pca(dir_input_path, dir_output_path_me, STRATEGY_V4_POST_L7_AVG, modality, filter_in_name=filter_in_name, overwrite=True)
    
    # dir_output_path_me = os.path.join(dir_output_path, "STRATEGY_V4_POST_L10_AVG")
    # exec_emb_and_pca(dir_input_path, dir_output_path_me, STRATEGY_V4_POST_L10_AVG, modality, filter_in_name=filter_in_name)
    
    # dir_output_path_me = os.path.join(dir_output_path, "STRATEGY_V4_POST_LALL_AVG")
    # exec_emb_and_pca(dir_input_path, dir_output_path_me, STRATEGY_V4_POST_LALL_AVG, modality, filter_in_name=filter_in_name)
    
    # dir_output_path_me = os.path.join(dir_output_path, "STRATEGY_V4_IMG_PLUS1")
    # exec_emb_and_pca(dir_input_path, dir_output_path_me, STRATEGY_V4_IMG_PLUS1, modality, filter_in_name=filter_in_name)
    
    
    # dir_output_path_me = os.path.join(dir_output_path, "STRATEGY_V3_FP_LANG_NORM_7")
    # exec_emb_and_pca(dir_input_path, dir_output_path_me, STRATEGY_V3_FP_LANG_NORM_7, modality, filter_in_name=filter_in_name)
    
    # dir_output_path_me = os.path.join(dir_output_path, "STRATEGY_V3_MP_POST_L4_AVG")
    # exec_emb_and_pca(dir_input_path, dir_output_path_me, STRATEGY_V3_MP_POST_L4_AVG, modality, filter_in_name=filter_in_name)
    

    #STRATEGY_V2_LANG_NORM_1
    # dir_output_path_me = os.path.join(dir_output_path, "STRATEGY_V2_LANG_NORM_1")
    # exec_emb_and_pca(dir_input_path, dir_output_path_me, STRATEGY_V2_LANG_NORM_1, modality, filter_in_name=filter_in_name)

    # #STRATEGY_V2_LANG_NORM_7
    # dir_output_path = os.path.join(dir_output_path, "STRATEGY_V2_LANG_NORM_7")
    # exec_emb_and_pca(dir_input_path, dir_output_path, STRATEGY_V2_LANG_NORM_7, modality, filter_in_name=filter_in_name)

    # dir_output_path_l5_avg = os.path.join(dir_output_path, "STRATEGY_V2_PRE_LAST5_AVG")
    # exec_emb_and_pca(dir_input_path, dir_output_path_l5_avg, STRATEGY_V2_PRE_LAST5_AVG, modality, filter_in_name=filter_in_name)

    # dir_output_path_l5 = os.path.join(dir_output_path, "STRATEGY_V2_PRE_LAST5")
    # exec_emb_and_pca(dir_input_path, dir_output_path_l5, STRATEGY_V2_PRE_LAST5, modality, filter_in_name=filter_in_name)

    #STRATEGY_V2_LANG_NORM_7
    # dir_output_path = os.path.join(dir_output_path, "STRATEGY_V2_LANG_NORM_AVG_PRE")
    # exec_emb_and_pca(dir_input_path, dir_output_path, STRATEGY_V2_LANG_NORM_AVG_PRE, modality, filter_in_name=filter_in_name)

    # #STRATEGY_V2_LANG_NORM_7
    # dir_output_path_me = os.path.join(dir_output_path, "STRATEGY_V2_LANG_NORM_I1")
    # exec_emb_and_pca(dir_input_path, dir_output_path_me, STRATEGY_V2_LANG_NORM_I1, modality, filter_in_name=filter_in_name)

    # #STRATEGY_V2_LANG_NORM_7
    # dir_output_path = os.path.join(dir_output_path, "STRATEGY_V2_LANG_NORM_FIRST")
    # exec_emb_and_pca(dir_input_path, dir_output_path, STRATEGY_V2_LANG_NORM_FIRST, modality, filter_in_name=filter_in_name)

    # #STRATEGY_V2_LANG_NORM_7
    # dir_output_path_me = os.path.join(dir_output_path, "STRATEGY_V2_POST_IMG")
    # exec_emb_and_pca(dir_input_path, dir_output_path_me, STRATEGY_V2_POST_IMG, modality, filter_in_name=filter_in_name)

    # #STRATEGY_V2_LANG_NORM_7
    # dir_output_path_me = os.path.join(dir_output_path, "STRATEGY_V2_IMG8A")
    # exec_emb_and_pca(dir_input_path, dir_output_path_me, STRATEGY_V2_IMG8A, modality, filter_in_name=filter_in_name)

    # dir_output_path = os.path.join(dir_output_path, "STRATEGY_V4_POST_L12_L10_AVG")
    # exec_emb_and_pca(dir_input_path, dir_output_path, STRATEGY_V4_POST_L12_L10_AVG, modality, filter_in_name=filter_in_name)
    
    # dir_output_path = os.path.join(dir_output_path, "STRATEGY_V4_POST_5Layer_L10_AVG")
    # exec_emb_and_pca(dir_input_path, dir_output_path, STRATEGY_V4_POST_5Layer_L10_AVG, modality, filter_in_name=filter_in_name)

    # dir_output_path = os.path.join(dir_output_path, "STRATEGY_V2_VISION_NORM_AVG")
    # exec_emb_and_pca(dir_input_path, dir_output_path, STRATEGY_V2_VISION_NORM_AVG, modality, filter_in_name=filter_in_name)

    # dir_output_path = os.path.join(dir_output_path, "STRATEGY_V2_POST_LAST")
    # exec_emb_and_pca(dir_input_path, dir_output_path, STRATEGY_V2_POST_LAST, modality, filter_in_name=filter_in_name)

    # dir_output_path_40 = os.path.join(dir_output_path, "STRATEGY_V2_POST_L4_LAST_40")
    # exec_emb_and_pca(dir_input_path, dir_output_path_40, STRATEGY_V2_POST_L4_LAST, modality, filter_in_name=filter_in_name)

    # dir_output_path_i1 = os.path.join(dir_output_path, "STRATEGY_V2_LANG_NORM_I1")
    # exec_emb_and_pca(dir_input_path, dir_output_path_i1, STRATEGY_V2_LANG_NORM_I1, modality, filter_in_name=filter_in_name)

    # dir_output_path_i2 = os.path.join(dir_output_path, "STRATEGY_V2_LANG_NORM_I2")
    # exec_emb_and_pca(dir_input_path, dir_output_path_i2, STRATEGY_V2_LANG_NORM_I2, modality, filter_in_name=filter_in_name)

    # dir_output_path_pre_last = os.path.join(dir_output_path, "STRATEGY_V2_PRE_LAST")
    # exec_emb_and_pca(dir_input_path, dir_output_path_pre_last, STRATEGY_V2_PRE_LAST, modality, filter_in_name=filter_in_name)

    # # # # STRATEGY_LANG_NORM_3
    # dir_output_path = os.path.join(out_dir,  "STRATEGY_LANG_NORM_3")
    # #exec_emb_and_pca(dir_input_path, dir_output_path, STRATEGY_LANG_NORM_3, modality)

    # # # # STRATEGY_LANG_NORM_7
    # dir_output_path = os.path.join(out_dir,  "STRATEGY_LANG_NORM_7")
    # exec_emb_and_pca(dir_input_path, dir_output_path, STRATEGY_LANG_NORM_7, modality)

    # # # # STRATEGY_LANG_NORM_10
    # dir_output_path = os.path.join(out_dir,  "STRATEGY_LANG_NORM_10")
    # exec_emb_and_pca(dir_input_path, dir_output_path, STRATEGY_LANG_NORM_10, modality)

    # # # STRATEGY_LANG_NORM_7_AVG
    # dir_output_path = os.path.join(out_dir,  "STRATEGY_LANG_NORM_7_AVG")
    # exec_emb_and_pca(dir_input_path, dir_output_path, STRATEGY_LANG_NORM_7_AVG, modality, pca_only=True, overwrite=True)

    #STRATEGY_LN7_4_12_NORM_VN_NORM
    # dir_output_path = os.path.join(out_dir,  "STRATEGY_LN7_4_12_NORM_VN_NORM")
    # exec_emb_and_pca(dir_input_path, dir_output_path, STRATEGY_LN7_4_12_NORM_VN_NORM, modality, pca_only=True, pca_only_750=True)

    # # # # STRATEGY_VISION_NORM
    # dir_output_path = os.path.join(out_dir,  "STRATEGY_VISION_NORM")
    # exec_emb_and_pca(dir_input_path, dir_output_path, STRATEGY_VISION_NORM, modality)

    # # # # STRATEGY_VISION_2_5_10_17_NORM
    # dir_output_path = os.path.join(out_dir, "STRATEGY_VISION_2_5_10_17_NORM")
    # exec_emb_and_pca(dir_input_path, dir_output_path, STRATEGY_VISION_2_5_10_17_NORM, modality)


    # # STRATEGY_LN_1_VN
    # dir_output_path = os.path.join(out_dir,  "STRATEGY_LN_1_VN")
    # exec_emb_and_pca(dir_input_path, dir_output_path, STRATEGY_LN_1_VN, modality)

    # # STRATEGY_LN_3_VN
    # dir_output_path = os.path.join(out_dir,  "STRATEGY_LN_3_VN")
    # exec_emb_and_pca(dir_input_path, dir_output_path, STRATEGY_LN_3_VN, modality)

    # # STRATEGY_LANG_4_12_NORM
    # dir_output_path = os.path.join(out_dir, "STRATEGY_LANG_4_12_NORM")
    # exec_emb_and_pca(dir_input_path, dir_output_path, STRATEGY_LANG_4_12_NORM, modality)

    # # # STRATEGY_VISION_23
    # dir_output_path = os.path.join(out_dir,  "STRATEGY_VISION_23")
    # exec_emb_and_pca(dir_input_path, dir_output_path, STRATEGY_VISION_23, modality)

    # get_stim_id_list(dir_input_path)
    # compute_tr_upper(dir_input_path, stim_id, layer_name)
    # modality = 'visual'
    # infolder = os.path.join(utils.get_raw_data_dir(), modality)
    # outfile = os.path.join(utils.get_pca_dir(), 'friends_movie10', modality, 'features_train_new_no_pca3.npy')
    # #features_combined_npy(infolder, outfile, modality, True, True)
    # #extract_raw_visual_features()
    # #extract_raw_audio_features()
    # #extract_raw_language_features()
    # #do_pca('language')

    # modality = 'visual'
    # inpath = os.path.join(utils.get_stimulus_pre_features_dir(),'raw_fit', modality)
    # outfile = os.path.join(utils.get_pca_dir(), 'friends_movie10', modality, 'features__r50_ft_1000.npy')
    # do_pca(inpath, outfile, modality, do_zscore=True,skip_pca_just_comgine=True, n_components=1000)



    #extract_preprocessed_video_content()
    # model_name = 'lora-20-distributed-s15'
    # custom_filter = "friends_s02*.h5"
    # extract_raw_visual_features_r50_ft(model_name, custom_filter)
    # custom_filter = "friends_s04*.h5"
    # extract_raw_visual_features_r50_ft(model_name, custom_filter)
    # custom_filter = "friends_s06*.h5"
    # extract_raw_visual_features_r50_ft(model_name, custom_filter)
    #extract_raw_visual_features_r50_ft()
    #print(inpath)
    #print(outfile)
    #extract_preprocessed_video_content()
    #extract_raw_visual_features_from_preprocessed_video()
    # save_dir_temp = utils.get_tmp_dir()
    # tr = 1.49
    # transform = define_frames_transform()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # filename = '/home/bagga005/algo/comp_data/stimulus_features/pre/visual/friends_s02e01a.h5'
    # outfile = '/home/bagga005/algo/comp_data/stimulus_features/raw/visual/friends_s02e01a.h5'
    #extract_visual_features_r50_ft(filename, device, outfile, "friends_s02e01a")
    #extract_raw_visual_features_r50_ft()
    #extract_raw_visual_features()



