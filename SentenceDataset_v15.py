import utils
from glob import glob
from torch.utils.data import Dataset
import string, re
import csv
import ast


def normalize_pauses(text):
    return re.sub(r'\.{3,8}', '\n', re.sub(r'\.{9,}', '\n\n', text))

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
    #print(f"Loading {file_path}")
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
                    dialogue_str = row['dialogue_per_tr'].strip()
                    if dialogue_str:
                        dialogue_per_str = ast.literal_eval(dialogue_str)
                    else:
                        dialogue_per_str = []
                    transcript_row['dialogue_per_tr'] = dialogue_per_str
                    # print(f"dialogue_per_tr: {transcript_row['dialogue_per_tr']}, length: {len(transcript_row['dialogue_per_tr'])}")
                    # print(f"words_per_tr: {transcript_row['words_per_tr']}, length: {len(transcript_row['words_per_tr'])}")
                    assert len(transcript_row['dialogue_per_tr']) == len(transcript_row['words_per_tr']), f"Length mismatch: words_per_tr={len(transcript_row['words_per_tr'])}, dialogue_per_tr={len(transcript_row['dialogue_per_tr'])} in row {row_num}"
                transcript_data.append(transcript_row)
                
            except (ValueError, SyntaxError) as e:
                print(f"Parsing error at row {row_num}: {e}")
                print(f"Problematic data:")
                print(f"  words_per_tr: '{row['words_per_tr']}'")
                print(f"  onsets_per_tr: '{row['onsets_per_tr']}'")
                print(f"  durations_per_tr: '{row['durations_per_tr']}'")
                raise
    
    return transcript_data

def load_all_tsv_for_one_episode(stim_id, isEnhanced=False):
    root_data_dir = utils.get_data_root_dir()
    #load transcript files
    if isEnhanced:
        t_files = glob(f"{root_data_dir}/algonauts_2025.competitors/stimuli/transcripts/friends/s*/enhanced/{stim_id}*.tsv")
    else:
        t_files = glob(f"{root_data_dir}/algonauts_2025.competitors/stimuli/transcripts/*/*/{stim_id}*.tsv")
    t_files.sort()
    f_stimuli = {f.split("/")[-1].split(".")[0]: f for f in t_files}
    print(f_stimuli)
    
    assert len(f_stimuli) > 0, "transcript file not found"
    trans_iterator = enumerate(f_stimuli.items())
    trans_info_list = []
    transcript_data = None
    isfirst = True
    for j, (trans_id, trans_path) in trans_iterator:
        #print(f"Handling {trans_id}", trans_path)
        tr_lines = load_transcript_tsv(trans_path, isEnhanced)
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


def get_full_transcript(stim_id):
    stim_id = utils.get_full_transcript_id(stim_id)
    stim_iden = stim_id[:-1]
    if stim_iden[-1] == '0' or stim_iden[-1] == '1':
        stim_iden = stim_iden[:-1]

    transcript_data, trans_info_list, total_tr_len = load_all_tsv_for_one_episode(stim_iden, isEnhanced=False)
    tr_start = 0
    tr_length =0
    foundMatch = False
    for tr_info in trans_info_list:
        if tr_info['trans_id'] == stim_id:
            tr_length = tr_info['len']
            foundMatch = True
            break
        else:
            tr_start += tr_info['len']
    
    assert foundMatch, "Transcript file not found for stimulus"
    return transcript_data, tr_start, tr_length

class SentenceDataset_v15(Dataset):
    def __init__(self, transcript_id, n_used_words=1000, prep_sentences="contpretr-friends-v1"):
        transcript_data, self.tr_start, self.tr_length = get_full_transcript(transcript_id)
        self.sentences = [item['text_per_tr'] for item in transcript_data]
        self.prep_sentences = prep_sentences
        if self.prep_sentences=="contpretr-friends-v1":
            self.sentences = [s if s != '' else "..." for s in self.sentences]
        #print('total len',len(self.sentences))
        #print('start',self.tr_start, 'self.tr_length',self.tr_length)
        self.n_used_words = n_used_words

    def __len__(self):
        return self.tr_length

    def __getitem__(self, idx):
        text = {}

        effective_idx = idx + self.tr_start
        #print(effective_idx)
        text['fancy_post'] = self.sentences[effective_idx]
        postLen = len(text['fancy_post'].split())
        full_past_text = "".join(self.sentences[:effective_idx])
        #nopunct_text = tr_text#tr_text.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
        pre_words_quota = self.n_used_words - postLen
        text['fancy_pre']= " ".join(full_past_text.split(" ")[-pre_words_quota:])

        if self.prep_sentences=="contpretr-friends-v1":
            text['fancy_post'] = normalize_pauses(text['fancy_post']).rstrip(" ")
            text['fancy_pre'] = normalize_pauses(text['fancy_pre']).rstrip(" ")

        if text=="": text= " "
        return text
    
def get_transcript_dataSet_simple(stim_id, n_used_words=1000):
    root_data_dir = utils.get_data_root_dir()
    ds = SentenceDataset_v15(stim_id, n_used_words=n_used_words)
    return ds

#ds = get_transcript_dataSet_simple("life05")
#print(ds[10])