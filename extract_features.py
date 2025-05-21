import utils
import gzip
import pickle
import os
from glob import glob
from tqdm import tqdm
import h5py
import torch
import numpy as np
from model_r50_ft import VisionR50FineTuneModel
USE_LIGHTNING = True
# try:
#     from lightning.data import map
# except:
#     print("lightning.data not found, using multiprocessing")
#     USE_LIGHTNING = False
#     from multiprocessing import Pool
from algonaut_funcs import extract_visual_features_r50_ft, extract_visual_features_from_preprocessed_video, load_features, preprocess_features, extract_visual_preprocessed_features, perform_pca, extract_visual_features, get_vision_model, extract_audio_features, get_language_model, define_frames_transform, extract_language_features
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
    file_in_filter = ''
    exclude_list = []#['friends_s03e05b', 'friends_s03e06a']
    files = glob(f"{root_data_dir}/algonauts_2025.competitors/stimuli/movies/movie10/**/*.mkv")
    if file_in_filter:
        files = [f for f in files if file_in_filter in f]
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
    

def do_pca(inpath, outfile,modality, do_zscore=True,skip_pca_just_comgine=False, n_components = 250):
    root_data_dir = utils.get_data_root_dir()
    out_data_dir = utils.get_output_dir()
    
    files = glob(f"{inpath}/*.h5")
    # filter_in_name =''
    # if filter_in_name != '':
    #     files = [f for f in files if filter_in_name in f]
    # filter_out_name = 's07e'
    # if filter_out_name != '':
    #     files = [f for f in files if filter_out_name not in f]
    # filter_out_name2 = 'bourne'
    # if filter_out_name2 != '':
    #     files = [f for f in files if filter_out_name2 not in f]
    files.sort()
    print(len(files), files[:3], files[-3:])
    stimuli = {f.split("/")[-1].split(".")[0]: f for f in files}
    print(len(stimuli), list(stimuli)[:3], list(stimuli)[-3:])
    #fileFilter = "movie10_life02"
    boundary = []
    iterator = tqdm(enumerate(stimuli.items()), total=len(list(stimuli)))
    valdict = {}
    features = []
    for i, (stim_id, stim_path) in iterator:
        print(f"pca features for {stim_id}", stim_path)
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

        # Perform PCA
        features_pca = perform_pca(prepr_features, n_components, modality)
        print('pca features.shape', features_pca.shape)
    else:
        features_pca = features

    # slice out results
    from_idx =0
    for stim_id, size in boundary:
        if from_idx < 3000: print(stim_id, size)
        slice = features_pca[from_idx:from_idx+size,:]
        valdict[get_shortstim_name(stim_id)] = slice
        if from_idx < 3000: print(from_idx, from_idx + size)
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

def segment_to_extract(loaded_tensor, combine_strategy):
    if combine_strategy == COMBINE_STRATEGY_LAST:
        ten = loaded_tensor[-1,:]
    elif combine_strategy == COMBINE_STRATEGY_LAST3:
        #print('loaded_tensor.shape', loaded_tensor.shape)
        ten = loaded_tensor[-3:,:].reshape(1, -1)
        #print('ten.shape', ten.shape)
    elif combine_strategy == COMBINE_STRATEGY_FIRST:
        #print('loaded_tensor.shape', loaded_tensor.shape)
        ten = loaded_tensor[0,:]
        #print('ten.shape', ten.shape)
    else:
        raise ValueError(f"Invalid strategy: {combine_strategy}")
    
    #print(f"segment_to_extract: {combine_strategy}, ten.shape", ten.shape)
    return ten.flatten()

def get_stim_id_list(dir_path, filter_in_name=None):
    files = glob(f"{dir_path}/*_metadata.json")
    f_list = [f.split("/")[-1].split("_")[0] + "_" + f.split("/")[-1].split("_")[1] for f in files]
    f_list = list(set(f_list))

    if filter_in_name is not None:
        # Keep files that contain any of the strings in filter_in_name
        f_list = [f for f in f_list if any(filter_str in f for filter_str in filter_in_name)]

    return f_list

#LLM Only
STRATEGY_LANG_NORM_1 = 0
STRATEGY_LANG_NORM_3 = 1
STRATEGY_LANG_4_12_NORM = 2
#Vision 
STRATEGY_VISION_NORM = 10
STRATEGY_VISION_23= 11
#LLM + Vision
STRATEGY_LN_1_VN = 20

COMBINE_STRATEGY_LAST = 'last'
COMBINE_STRATEGY_LAST3 = 'last3'
COMBINE_STRATEGY_FIRST = 'first'
def save_combined_vlm_features(dir_input_path, dir_output_path, strategy, modality, filter_in_name=None):
    stim_id_list = get_stim_id_list(dir_input_path, filter_in_name)
    for stim_id in stim_id_list:
        print('stim_id', stim_id)
        features = []
        if strategy == STRATEGY_LANG_NORM_1:
            ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_LAST)
        elif strategy == STRATEGY_LANG_NORM_3:
            ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_LAST3)
        elif strategy == STRATEGY_VISION_NORM:
            ten1 = combine_vlm_features(dir_input_path, stim_id, "vision_model", COMBINE_STRATEGY_FIRST)
        elif strategy == STRATEGY_LN_1_VN:
            ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_LAST)
            ten2 = combine_vlm_features(dir_input_path, stim_id, "vision_model", COMBINE_STRATEGY_FIRST)
            ten1 = torch.cat((ten1, ten2), dim=1)
            #print('combined_tensor.shape', ten1.shape)
        elif strategy == STRATEGY_LANG_4_12_NORM:
            ten0 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_layers_4", COMBINE_STRATEGY_LAST)
            ten1 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_layers_12", COMBINE_STRATEGY_LAST)
            ten2 = combine_vlm_features(dir_input_path, stim_id, "language_model_model_norm", COMBINE_STRATEGY_LAST)
            ten1 = torch.cat((ten0, ten1, ten2), dim=1)
            #print('combined_tensor.shape', ten1.shape)
        else:
            raise ValueError(f"Invalid strategy: {strategy}")
        if ten1.dtype == torch.bfloat16:
            ten1 = ten1.to(torch.float32)
        visual_features = ten1.cpu().numpy()
        save_file = os.path.join(dir_output_path, f"{stim_id}.h5")
        with h5py.File(save_file, 'w') as f:
            group = f.create_group(stim_id)
            group.create_dataset(modality, data=visual_features, dtype=np.float32)   
        
def combine_vlm_features(dir_path, stim_id, layer_name, strategy):
    tensor_list = []
    tr_upper = compute_tr_upper(dir_path, stim_id, layer_name)
    for tr_i in range(tr_upper):
        file_path = os.path.join(dir_path, f"{stim_id}_tr_{tr_i}_{layer_name}.pt.gz")
        with gzip.open(file_path, 'rb') as f:
            loaded_tensor = pickle.load(f)
        tensor_list.append(segment_to_extract(loaded_tensor, strategy))
    combined_tensor = torch.stack(tensor_list, dim=0)
    #print(f"combine_features: {strategy}, combined_tensor.shape", combined_tensor.shape, "stim_id", stim_id)
    return combined_tensor

if __name__ == "__main__":
    out_dir = utils.get_output_dir()
    dir_input_path = os.path.join(out_dir, "embeddings")
    filter_in_name = ["s03", "s02", "s06"]

    # STRATEGY_LANG_NORM_1
    # dir_output_path = os.path.join(out_dir, "embeddings_combined", "STRATEGY_LANG_NORM_1")
    # strategy = STRATEGY_LANG_NORM_1
    # save_combined_vlm_features(dir_input_path, dir_output_path, strategy, "visual")
    # do_pca(dir_output_path, dir_output_path + "/features_train.npy", "visual", do_zscore=False, skip_pca_just_comgine=True)

    # # STRATEGY_LANG_NORM_3
    # dir_output_path = os.path.join(out_dir, "embeddings_combined", "STRATEGY_LANG_NORM_3")
    # strategy = STRATEGY_LANG_NORM_3
    # save_combined_vlm_features(dir_input_path, dir_output_path, strategy, "visual")
    # do_pca(dir_output_path, dir_output_path + "/features_train.npy", "visual", do_zscore=False, skip_pca_just_comgine=True)

    # # STRATEGY_VISION_NORM
    # dir_output_path = os.path.join(out_dir, "embeddings_combined", "STRATEGY_VISION_NORM")
    # strategy = STRATEGY_VISION_NORM
    # save_combined_vlm_features(dir_input_path, dir_output_path, strategy, "visual")
    # do_pca(dir_output_path, dir_output_path + "/features_train.npy", "visual", do_zscore=False, skip_pca_just_comgine=True)

    # STRATEGY_LN_1_VN
    # dir_output_path = os.path.join(out_dir, "embeddings_combined", "STRATEGY_LN_1_VN")
    # os.makedirs(dir_output_path, exist_ok=True)	
    # strategy = STRATEGY_LN_1_VN
    # save_combined_vlm_features(dir_input_path, dir_output_path, strategy, "visual")
    # print(f'**Starting pca for STRATEGY_LN_1_VN') 
    # do_pca(dir_output_path, dir_output_path + "/features_train.npy", "visual", do_zscore=False, skip_pca_just_comgine=True)

    # STRATEGY_LANG_4_12_NORM
    dir_output_path = os.path.join(out_dir, "embeddings_combined", "STRATEGY_LANG_4_12_NORM")
    os.makedirs(dir_output_path, exist_ok=True)	
    strategy = STRATEGY_LANG_4_12_NORM
    save_combined_vlm_features(dir_input_path, dir_output_path, strategy, "visual", filter_in_name=filter_in_name)
    print(f'**Starting pca for STRATEGY_LANG_4_12_NORM') 
    do_pca(dir_output_path, dir_output_path + "/features_train.npy", "visual", do_zscore=False, skip_pca_just_comgine=True)

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
