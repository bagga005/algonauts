from train import load_stimulus_features_friends_s7, align_features_and_fmri_samples_friends_s7, get_features, align_features_and_fmri_samples
import utils
import zipfile
import os
from glob import glob
import numpy as np
from algonaut_funcs import prepare_s7_fmri_for_alignment

# #dummy np
# dummy_np = np.zeros((1000, 1000))
# utils.save_predictions_accuracy(1, dummy_np, None)
# exit()

FORMAT_CODA = 1
FORMAT_FLAT = 2
FORMAT_WITH_MODALITY =3

def get_dict_key_for_subject(sub):
    return f'sub-0{sub}'

def append_to_dict(dict, subject, stimuli_id, data, format, modality):
    if format == FORMAT_CODA:
        dict[get_dict_key_for_subject(subject)][stimuli_id] = data
    elif format == FORMAT_FLAT:
        dict[stimuli_id] = data
    elif format == FORMAT_WITH_MODALITY:
        dict[modality][stimuli_id] = data

def init_dict(dict, subject, format, modality):
    if format == FORMAT_CODA:
        dict = {}
        for sub in subjects:
            dict[get_dict_key_for_subject(sub)] = {}
    elif format == FORMAT_FLAT:
        dict = {}
    elif format == FORMAT_WITH_MODALITY:
        dict[modality] = {}
    return dict

def preppare_output_files(subjects, exp_name, format=FORMAT_CODA, modality='language', movie_name='friends-s07'):
    submission_predictions = init_dict({}, subjects, format, modality)
    pads = np.zeros((5,1000))

    for sub in subjects:
        fmri, boundary = prepare_s7_fmri_for_alignment(sub)
        predictions_file = utils.get_predictions_file_path(sub, "friends-s07")
        sub_predictions = np.load(predictions_file, allow_pickle=True)
        predictions_dict = {}
        from_idx = 0
        total_size =0
        num_stimuli =0
        for stim_id, size in boundary:
            num_stimuli +=1
            total_size += size
            effective_size = size-10
            slice = sub_predictions[from_idx:from_idx+effective_size,:]
            slice = np.concatenate((pads, slice, pads), axis=0)
            append_to_dict(submission_predictions, sub, stim_id, slice.astype(np.float32), format, modality)
            #if from_idx < 3000: print(from_idx, from_idx + size)
            from_idx = from_idx + effective_size
            assert slice.shape[0] == size, f"size mismatch while slicing {stim_id} {slice.shape[0]} {size}"
        assert total_size == (sub_predictions.shape[0] + num_stimuli*2*pads.shape[0]), f"total_size {total_size} != sub_predictions.shape[0] {sub_predictions.shape[0] + num_stimuli*2*pads.shape[0]}"
    print(submission_predictions.keys())
    # print(submission_predictions['sub-01'].keys())
    # print(submission_predictions['sub-01']['s07e01a'].shape)

    predictions_dir = os.path.join(utils.get_output_dir(), 'predictions', exp_name)
    file_name = f"fmri_predictions_{movie_name}"
    file_name_w_ext = f"{file_name}.npy"
    output_file = os.path.join(predictions_dir, file_name_w_ext)
    np.save(output_file, submission_predictions)
    print(f"Formatted predictions saved to: {output_file}")

    # Zip the saved file for submission
    zip_file = os.path.join(predictions_dir, f"{file_name}.zip")
    with zipfile.ZipFile(zip_file, 'w') as zipf:
        zipf.write(output_file, os.path.basename(output_file))
    print(f"Submission file successfully zipped as: {zip_file}")


if __name__ == "__main__":
    exp_name = utils.get_experiment_name()
    subjects = [1]
    movie_name = "friends-s07"
    preppare_output_files(subjects, exp_name, FORMAT_CODA, movie_name=movie_name)