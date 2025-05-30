# Example usage and test
from glob import glob
import utils
import csv
import ast
import transcripts_enhancer
import os

def load_transcript_tsv(file_path, isEnhanced=False):
    """
    Loads a TSV file containing transcript data with specific fields.
    
    Args:
        file_path (str): Path to the TSV file
        
    Returns:
        list: Array of objects with text_per_tr, words_per_tr, onsets_per_tr, durations_per_tr fields
        
    Raises:
        AssertionError: If validation rules are violated
    """
    
    transcript_data = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')
        
        # Check that all required fields are in the header
        if isEnhanced:
            required_fields = ['text_per_tr', 'words_per_tr', 'onsets_per_tr', 'durations_per_tr', 'dialogue_per_tr']
        else:
            required_fields = ['text_per_tr', 'words_per_tr', 'onsets_per_tr', 'durations_per_tr']

        for field in required_fields:
            assert field in reader.fieldnames, f"Required field '{field}' not found in TSV header"
        
        for row_num, row in enumerate(reader, start=2):  # Start at 2 since header is line 1
            # Assert that each row has all 4 fields
            for field in required_fields:
                assert field in row, f"Row {row_num}: Missing required field '{field}'"
            
            try:
                # Parse words_per_tr using ast.literal_eval to handle mixed quotes
                words_str = row['words_per_tr'].strip()
                if words_str:
                    words_per_tr = ast.literal_eval(words_str)
                else:
                    words_per_tr = []
                
                # Parse onsets_per_tr - these should be in JSON format
                onsets_str = row['onsets_per_tr'].strip()
                if onsets_str:
                    onsets_list = ast.literal_eval(onsets_str)
                    onsets_per_tr = [float(x) for x in onsets_list]
                else:
                    onsets_per_tr = []
                
                # Parse durations_per_tr - these should be in JSON format  
                durations_str = row['durations_per_tr'].strip()
                if durations_str:
                    durations_list = ast.literal_eval(durations_str)
                    durations_per_tr = [float(x) for x in durations_list]
                else:
                    durations_per_tr = []
                
                transcript_row = {
                    'text_per_tr': row['text_per_tr'],  # Can be empty
                    'words_per_tr': words_per_tr,
                    'onsets_per_tr': onsets_per_tr,
                    'durations_per_tr': durations_per_tr
                }
                if isEnhanced:
                    transcript_row['dialogue_per_tr'] = row['dialogue_per_tr']
                    assert len(transcript_row['dialogue_per_tr']) == len(transcript_row['words_per_tr']), f"Length mismatch: words_per_tr={len(transcript_row['words_per_tr'])}, dialogue_per_tr={len(transcript_row['dialogue_per_tr'])}"
                transcript_data.append(transcript_row)
                
            except (ValueError, SyntaxError) as e:
                print(f"Parsing error at row {row_num}: {e}")
                print(f"Problematic data:")
                print(f"  words_per_tr: '{row['words_per_tr']}'")
                print(f"  onsets_per_tr: '{row['onsets_per_tr']}'")
                print(f"  durations_per_tr: '{row['durations_per_tr']}'")
                raise
    
    return transcript_data

def save_transcript_with_dialogue_tsv(file_path, transcript_data):
    """
    Saves transcript data with dialog information to a TSV file.
    
    Args:
        file_path (str): Path where the TSV file will be saved
        transcript_data (list): Array of objects with 5 fields:
                               text_per_tr, words_per_tr, onsets_per_tr, durations_per_tr, dialogue_per_tr
    """
    directory_path = os.path.dirname(file_path)
    os.makedirs(directory_path, exist_ok=True)
    fieldnames = ['text_per_tr', 'words_per_tr', 'onsets_per_tr', 'durations_per_tr', 'dialogue_per_tr']
    
    with open(file_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter='\t')
        
        # Write header
        writer.writeheader()
        
        # Write data rows
        for row_data in transcript_data:
            # Convert arrays to Python literal strings (maintains mixed quote format)
            tsv_row = {
                'text_per_tr': row_data['text_per_tr'],
                'words_per_tr': repr(row_data['words_per_tr']),
                'onsets_per_tr': repr(row_data['onsets_per_tr']),
                'durations_per_tr': repr(row_data['durations_per_tr']),
                'dialogue_per_tr': repr(row_data['dialogue_per_tr'])
            }
            writer.writerow(tsv_row)
def print_key_stats(key_stats_list):
    per_skipped_p1_total = 0
    per_skipped_p2_total = 0
    total_dialogues_total = 0
    total_match_rate_dialogue_total = 0
    total_match_rate_transcript_total = 0
    num_stats = len(key_stats_list)
    for key_stats in key_stats_list:
        total_dialogues, num_skipped_p1, num_skipped_p2, match_rate_dialogue, match_rate_transcript = key_stats["key_stats"]
        per_skipped_p1 = num_skipped_p1 / total_dialogues if total_dialogues > 0 else 0
        per_skipped_p2 = num_skipped_p2 / total_dialogues if total_dialogues > 0 else 0
        print(f'{key_stats["stim_id"]}: {key_stats["key_stats"]} (p1: {per_skipped_p1*100:.2f}%, p2: {per_skipped_p2*100:.2f}%), {total_dialogues} dialogues, {num_skipped_p1} skipped in p1, {num_skipped_p2} skipped in p2, {match_rate_dialogue*100:.2f}% dialogue match rate, {match_rate_transcript*100:.2f}% transcript match rate')
        per_skipped_p1_total += num_skipped_p1
        per_skipped_p2_total += num_skipped_p2
        total_match_rate_dialogue_total += match_rate_dialogue
        total_match_rate_transcript_total += match_rate_transcript
        total_dialogues_total += total_dialogues
    print(f'Average of per_skipped_p1: {per_skipped_p1_total / total_dialogues_total*100:.2f}%, Average of per_skipped_p2: {per_skipped_p2_total / total_dialogues_total*100:.2f}%', f'Total dialogues: {total_dialogues_total}', f'Total skipped in p1: {per_skipped_p1_total}', f'Total skipped in p2: {per_skipped_p2_total}', f'Total match rate dialogue: {total_match_rate_dialogue_total / num_stats:.2f}%', f'Total match rate transcript: {total_match_rate_transcript_total /num_stats:.2f}%')

def load_all_tsv_for_one_episode(stim_id):
    root_data_dir = utils.get_data_root_dir()
    #load transcript files
    t_files = glob(f"{root_data_dir}/stimuli/transcripts/friends/s*/{stim_id}*.tsv")
    t_files.sort()
    f_stimuli = {f.split("/")[-1].split(".")[0]: f for f in t_files}
    print(len(f_stimuli), list(f_stimuli)[:3], list(f_stimuli)[-3:])
    

    trans_iterator = enumerate(f_stimuli.items())
    trans_info_list = []
    transcript_data = None
    isfirst = True
    for j, (trans_id, trans_path) in trans_iterator:
        print(f"Handling {trans_id}", trans_path)
        tr_lines = load_transcript_tsv(trans_path)
        num_lines = len(tr_lines)
        if isfirst:
            transcript_data = tr_lines
            isfirst = False
        else:
            transcript_data.extend(tr_lines)
        trans_info = {"len": num_lines, "stimd_id": stim_id, "trans_id": trans_id, "trans_path": trans_path}
        trans_info_list.append(trans_info)
    total_tr_len = 0
    for trans_info in trans_info_list:
        total_tr_len += trans_info["len"]

    return transcript_data, trans_info_list, total_tr_len

def run_for_one_episode(stim_id, stim_path):
    transcript_data, trans_info_list, total_tr_len = load_all_tsv_for_one_episode(stim_id)
    #outfile_path
    season_id = 's' + stim_id[10]
    #enhance the transcripts
    #enhanced_transcript_data = transcript_data
    assert total_tr_len == len(transcript_data), f"Total transcript length {len(transcript_data)} does not match the number of transcript data {total_tr_len}"
    enhanced_transcript_data, key_stats = transcripts_enhancer.enhance_transcripts_v2(transcript_data, stim_path, run_phase_2=True)
    assert len(enhanced_transcript_data) == total_tr_len, f"Enhanced transcript length {len(enhanced_transcript_data)} does not match the number of transcript data {total_tr_len}"

    #save the enhanced transcripts
    counter = 0
    for trans_info in trans_info_list: 
        outfile_path = os.path.join(utils.get_data_root_dir(),'stimuli', 'transcripts', 'friends',season_id,'enhanced',f'{trans_info["trans_id"]}.tsv')
        print(f"Saving to {outfile_path}")
        file_enhanced_transcript_data = enhanced_transcript_data[counter:counter+trans_info["len"]]
        assert len(file_enhanced_transcript_data) == trans_info["len"], f"File enhanced transcript length {len(file_enhanced_transcript_data)} does not match the number of transcript data {trans_info['len']}"
        save_transcript_with_dialogue_tsv(outfile_path, file_enhanced_transcript_data)
        counter += trans_info["len"]
    return key_stats

def run_for_all_episodes(print_stats=True):
    root_data_dir = utils.get_data_root_dir()
    
    #list of full text transcripts
    file_in_filter = ''
    exclude_list = []#['friends_s03e05b', 'friends_s03e06a']
    files = glob(f"{root_data_dir}/stimuli/transcripts/friends/full/*.txt")
    updated_files = []
    for file in files:
        exclude_found = False
        for exclude in exclude_list:
            if exclude in file:
                exclude_found = True
                break
        if not exclude_found:
            updated_files.append(file)
    files = updated_files
    if file_in_filter:
        files = [f for f in files if file_in_filter in f]
    files.sort()

    stimuli = {f.split("/")[-1].split(".")[0]: f for f in files}
    print(len(stimuli), list(stimuli)[:3], list(stimuli)[-3:])


    key_stats_list = []
    videos_iterator = enumerate(stimuli.items())
    for i, (stim_id, stim_path) in videos_iterator:
        key_stats = run_for_one_episode(stim_id, stim_path)
        key_stats_list.append({"stim_id": stim_id, "key_stats": key_stats})

    if print_stats:
        print_key_stats(key_stats_list)

def test_with_1_episode(print_stats=True):
    stim_id = 'friends_s04e21'
    stim_path = os.path.join(utils.get_data_root_dir(), 'stimuli', 'transcripts', 'friends', 'full', f'{stim_id}.txt')
    key_stats_list = []
    key_stats = run_for_one_episode(stim_id, stim_path)
    key_stats_list.append({"stim_id": stim_id, "key_stats": key_stats})
    if print_stats:
        print_key_stats(key_stats_list)



if __name__ == "__main__":
    #run_for_all_episodes()
    test_with_1_episode()