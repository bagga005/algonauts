import string
import pandas as pd
from glob import glob
from torch.utils.data import Dataset, DataLoader
import utils
from Scenes_and_dialogues import get_scene_dialogue, get_dialogue_list, match_dialogues_to_transcript_data, \
    get_dialogue_display_text, get_scene_for_dialogue, get_scene_display_text, get_closest_dialogue_for_row, get_scene_and_dialogues_display_text, get_scenes_summary
from tabulate import tabulate
import string, re
import numpy as np
from transcripts_handler import load_all_tsv_for_one_episode
import os
from rapidfuzz import fuzz

def normalize_pauses(text):
    return re.sub(r'\.{3,8}', '\n', re.sub(r'\.{9,}', '\n\n', text))

class SentenceDataset(Dataset):
    def __init__(self, sentences, mode="last_n_trs", last_n_trs=5, n_used_words=510, prep_sentences=None):
        self.sentences = sentences
        self.prep_sentences = prep_sentences
        if self.prep_sentences=="contpretr-friends-v1":
            self.sentences = [s if not(s is np.nan) else "..." for s in self.sentences]

        self.mode=mode
        self.last_n_trs = last_n_trs;
        self.n_used_words = n_used_words;

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        text = ""
        if self.mode == "last_n_trs":
          text= self.sentences[idx-self.last_n_trs: idx+1]
          text= "".join(text)

        elif self.mode=="n_used_words":
          tr_text = "".join(self.sentences[:idx+1])
          nopunct_text = tr_text#tr_text.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
          text= " ".join(nopunct_text.split(" ")[-self.n_used_words:])

        if self.prep_sentences=="contpretr-friends-v1":
            text = normalize_pauses(text)

        if text=="": text= " "
        return text

class SentenceDataset_v15(Dataset):
    def __init__(self, transcript_id, n_used_words=1000, prep_sentences="contpretr-friends-v1"):
        transcript_data, self.tr_start, self.tr_length = get_full_transcript(transcript_id)
        self.sentences = [item['text_per_tr'] for item in transcript_data]
        self.prep_sentences = prep_sentences
        if self.prep_sentences=="contpretr-friends-v1":
            self.sentences = [s if not(s is np.nan) else "..." for s in self.sentences]

        self.n_used_words = n_used_words

    def __len__(self):
        return self.tr_length

    def __getitem__(self, idx):
        text = {}

        effective_idx = idx + self.tr_start
        print(f"effective_idx: {effective_idx} {idx} {self.tr_start}")
        text['fancy_post'] = self.sentences[effective_idx]
        postLen = len(text['fancy_post'].split())
        text['fancy_pre'] = "".join(self.sentences[:effective_idx])
        #nopunct_text = tr_text#tr_text.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
        pre_words_quota = self.n_used_words - postLen
        text['fancy_pre']= " ".join(text['fancy_pre'].split(" ")[-pre_words_quota:])

        if self.prep_sentences=="contpretr-friends-v1":
            text['fancy_post'] = normalize_pauses(text['fancy_post'])
            text['fancy_pre'] = normalize_pauses(text['fancy_pre'])
        print(f"text['fancy_pre']: {text['fancy_pre']}")
        print(f"text['fancy_post']: {text['fancy_post']}")

        if text=="": text= " "
        return text
    
def get_best_text(ds15, ds2, idx, skip_video_tokens=False, num_videos=8):
    text15 = ds15[idx]
    text2 = ds2[idx]

    npre = text2['normal_pre']
    npst = text2['normal_post']
    if not npst: npst = ""
    if npre:
        text2_normal = npre + ' '  +npst
    else:
        text2_normal = npst

    text2_normal = re.sub(r'\.{3,}', ' ', text2_normal)
    last_2_text2_normal = utils.get_last_x_words(text2_normal, 2)
    last_2_text2_normal_words_list = last_2_text2_normal.split()
    last_text15_word = utils.normalize_and_clean_word(utils.get_last_x_words(text15, 1))
    matched = False
    for word in last_2_text2_normal_words_list:
        score = fuzz.ratio(utils.normalize_and_clean_word(word), last_text15_word)
        if score > 80:
            matched = True
            break

    if matched:
        return combine_pre_post_text(text2, skip_video_tokens, num_videos), True
    else:
        return text15, False
            



# df = pd.read_csv(transcript_file, sep='\t').fillna("")
#         dataset = SentenceDataset(df["text_per_tr"].tolist(), mode="n_used_words", n_used_words=n_used_words, prep_sentences=prep_sentences)

def get_full_transcript(stim_id):
    transcript_data, trans_info_list, total_tr_len = load_all_tsv_for_one_episode(stim_id[:-1], isEnhanced=False)
    tr_start = 0
    tr_length =0
    for tr_info in trans_info_list:
        if tr_info['trans_id'] == stim_id:
            tr_length = tr_info['len']
            break
        else:
            tr_start += tr_info['len']
    print(f"tr_start: {tr_start}, tr_length: {tr_length}")
    return transcript_data, tr_start, tr_length

def get_transcript_dataSet_simple(stim_id, n_used_words=1000):
    root_data_dir = utils.get_data_root_dir()
    ds = SentenceDataset_v15(stim_id, n_used_words=n_used_words)
    return ds
    

def get_transcript_dataSet(stim_id, always_post_speaker=True, exclude_post_dialogue_separator=True, n_used_words=1000, skip_pre_post_split=False, use_summary=False, use_present_scene=False):
    root_data_dir = utils.get_data_root_dir()
    transcript_data, trans_info_list, total_tr_len = load_all_tsv_for_one_episode(stim_id[:-1], isEnhanced=True)
    tr_start = 0
    tr_length =0
    for tr_info in trans_info_list:
        if tr_info['trans_id'] == stim_id:
            tr_length = tr_info['len']
            break
        else:
            tr_start += tr_info['len']
    dialogue_file = os.path.join(root_data_dir, 'algonauts_2025.competitors','stimuli', 'transcripts', 'friends', 'full', f'{stim_id[:-1]}.txt')
    dialogues = get_scene_dialogue(dialogue_file)
    scene_summary_data = None
    if use_summary:
        scene_summary_data = get_scenes_summary(stim_id)
        if not scene_summary_data:
            raise Exception(f'{stim_id} summary not found')
    trans_dataset = SentenceDataset_v2(transcript_data, dialogues, tr_start, tr_length, always_post_speaker=always_post_speaker, \
        exclude_post_dialogue_separator=exclude_post_dialogue_separator, n_used_words=n_used_words, skip_pre_post_split=skip_pre_post_split, \
            use_summary=use_summary, scene_summary=scene_summary_data, use_present_scene=use_present_scene)
    return trans_dataset

def combine_pre_post_text(textData, skip_video_tokens=False, num_videos=8, mvl_pix_last=True):
    pre_text = textData['fancy_pre']
    post_text = textData['fancy_post']
    if mvl_pix_last:
        if skip_video_tokens:
            if pre_text:
                video_prefix = pre_text
            else:
                video_prefix = ''
        else:
            if pre_text:
                video_prefix = pre_text + ''.join([f'\nFrame{i+1}: <image>' for i in range(num_videos)])
            else:
                video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(num_videos)])
                video_prefix = video_prefix[:-1]
                            
        if post_text:
            if video_prefix:
                question_for_embeddings = video_prefix + "\n" + post_text
            else:
                question_for_embeddings = post_text
        else:
            question_for_embeddings = video_prefix 
    
    else:
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(num_videos)])
        video_prefix = video_prefix[:-1]
        if pre_text:
            total_text = pre_text
        else:
            total_text = ''
        #combine pre and post text
        if post_text:
            total_text = total_text + ("\n" if total_text else "") + post_text
            
        if total_text:
            question_for_embeddings = video_prefix + "\n" + total_text
        else:
            question_for_embeddings = video_prefix
        #utils.log_to_file('question_for_embeddings', question_for_embeddings)

            

            
    return question_for_embeddings

class SentenceDataset_v2(Dataset):
    def __init__(self, transcript_data, scene_and_dialogues, tr_start, length, n_used_words=1000, always_post_speaker=False, exclude_post_dialogue_separator=False, \
        skip_pre_post_split=False, use_summary=False, scene_summary=None, use_present_scene=False):
        self.scenes_and_dialogues = scene_and_dialogues
        self.dialogue_list = get_dialogue_list(scene_and_dialogues)
        self.transcript_data = transcript_data
        self.dialogue_list, self.text_to_position_map, self.position_to_text_map = match_dialogues_to_transcript_data(self.transcript_data, self.dialogue_list)
        self.tr_start = tr_start
        self.length = length
        self.n_used_words = n_used_words
        self.always_post_speaker = always_post_speaker
        self.exclude_post_dialogue_separator = exclude_post_dialogue_separator
        self.skip_pre_post_split = skip_pre_post_split
        self.scene_summary = scene_summary
        self.use_summary = use_summary
        self.use_present_scene = use_present_scene
        if use_summary and self.scene_summary is None:
            raise Exception(f'summary not found')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        
        if idx >= self.length:
            raise IndexError(f"Index {idx} is out of range for length {self.length}")
        effective_idx = idx + self.tr_start
        data = self.get_text_for_tr(effective_idx)
        return data
    
    def get_dialogues_for_row(self, row_idx):
        list_of_dialogues = []
        for dialogue in self.dialogue_list:
            if dialogue['matched_text_index_start'] != -1  and (dialogue['matched_row_index_start'] <= row_idx <= dialogue['matched_row_index_end']):
                list_of_dialogues.append(dialogue)
        #order dialogues by id
        list_of_dialogues.sort(key=lambda x: x['id'])
        return list_of_dialogues
    
    def get_closest_row_with_dialogue(self, row_idx):
        row_idx = row_idx - 1
        while row_idx > 0:
            if len(self.get_dialogues_for_row(row_idx)) > 0:
                return row_idx
            row_idx = row_idx - 1
        return -1
    

    def _get_text_from_dialogue_for_row(self, dialogue, row_idx, forcePostSpeaker=False):
        #debug info
        # #print(f"dialogue id: {dialogue['id']}")
        # #print(f"dialogue normalized text: {dialogue['normalized_text']}")
        # #print(f"dialogue length normalized text: {dialogue['length_normalized_text']}")
        # #print(f"dialogue matched row index start: {dialogue['matched_row_index_start']}")
        # #print(f"dialogue matched row index end: {dialogue['matched_row_index_end']}")
        # #print(f"dialogue matched text index start: {dialogue['matched_text_index_start']}")
        # #print(f"dialogue matched text index end: {dialogue['matched_text_index_end']}")
        # #print(f"row position to text map: {self.position_to_text_map.get((row_idx, 0))}")
        row_word_length = len(self.transcript_data[row_idx]['words_per_tr'])
        #print(f"_get_text_from_dialogue_for_row row_word_length: {row_word_length}")
        #is dialogue starting in this row
        starting_in_this_row = (dialogue['matched_row_index_start'] == row_idx) or self.skip_pre_post_split
        #is dialogue ending in this row
        ending_in_this_row = dialogue['matched_row_index_end'] == row_idx

        if dialogue["length_normalized_text"] == 1 or (starting_in_this_row and ending_in_this_row):
            #print(f"calling from 1")
            response_post = get_dialogue_display_text(dialogue, withSpeaker=True)
            start_index = 0
            response_pre = {
                    "fancy": None,
                    "normal": None
                }
        else:
            #we have to split this dialogue into multiple parts: middle and possibly pre and post
            #see how many words in the row overlap with the dialogue and which is the first and last word
            first_overlap_word_index = -1
            last_overlap_word_index = row_word_length - 1
            overlap_length = 0
            for word_idx in range(row_word_length):
                text_index = self.position_to_text_map.get((row_idx, word_idx))
                if (dialogue['matched_text_index_start'] <= text_index <= dialogue['matched_text_index_end']):
                    overlap_length += 1
                    if first_overlap_word_index == -1:
                        first_overlap_word_index = text_index
                    last_overlap_word_index = text_index
            
            #starting index of dialogue for this row
            offset = max(0, first_overlap_word_index -dialogue['matched_text_index_start'] )
            start_index = round(offset*dialogue['run_rate'])
            dialogue_length_for_row = round(dialogue['run_rate'] * overlap_length)
            
            if dialogue_length_for_row + start_index > len(dialogue['normalized_text']):
                dialogue_length_for_row = dialogue_length_for_row - 1
            
            #determine prefix and suffix for post text
            add_prefix_continuation_for_post = False
            add_suffix_continuation_for_post = False    
            
            #if we skip pre post split, we need to put pre + post part in post
            if self.skip_pre_post_split:
                dialogue_length_for_row = dialogue_length_for_row + start_index
                start_index = 0

            if start_index > 0:
                add_prefix_continuation_for_post = True
            if start_index + dialogue_length_for_row < len(dialogue['normalized_text']):
                add_suffix_continuation_for_post = True

            #prepare response object
            #print(f" calling for post get_dialogue_display_text start_index: {start_index}, length: {dialogue_length_for_row}")
            #print(f"calling from 2")

            

            response_post = get_dialogue_display_text(dialogue, withSpeaker=starting_in_this_row or forcePostSpeaker, start_index=start_index, length=dialogue_length_for_row,
                                                      add_prefix_continuation=add_prefix_continuation_for_post, add_suffix_continuation=add_suffix_continuation_for_post)

            #get pre text if needed
            if start_index > 0:
                #print(f" calling for pre get_dialogue_display_text start_index: {0}, length: {start_index }")
                #print(f"calling from 3")
                response_pre = get_dialogue_display_text(dialogue, withSpeaker=True, start_index=0, length=start_index, add_suffix_continuation=True )
            else:
                response_pre = {
                    "fancy": None,
                    "normal": None
                }

        response_object = {
            "fancy_post": response_post["fancy"],
            "normal_post": response_post["normal"],
            "fancy_pre":response_pre["fancy"],
            "normal_pre": response_pre["normal"],
        }
        
        return response_object
            
            



    def get_text_for_tr(self, row_idx):
        #print("!!!!!!!!!!starting for row: ", row_idx)
        fancy_post, normal_post, fancy_pre, normal_pre = None, None, None, None
        word_length = len(self.transcript_data[row_idx]['words_per_tr'])
        first_dialogue_in_row = None
        #setup post main text
        list_of_dialogues = self.get_dialogues_for_row(row_idx)
        for dialogue in list_of_dialogues:
            if first_dialogue_in_row is None:
                first_dialogue_in_row = dialogue
            forcePostSpeaker = normal_post is not None or self.always_post_speaker
            resp = self._get_text_from_dialogue_for_row(dialogue, row_idx, forcePostSpeaker = forcePostSpeaker)
            if resp['fancy_post']:
                if fancy_post is not None:
                    fancy_post = fancy_post + "\n" + resp['fancy_post']
                else:
                    fancy_post = resp['fancy_post']
            if resp['normal_post']:
                #print(f"resp['normal_post'] as not none: {resp['normal_post']}")
                if normal_post is not None:
                    normal_post = normal_post + "\n" + resp['normal_post']
                else:
                    normal_post = resp['normal_post']
            if resp['fancy_pre']:
                if fancy_pre is not None:
                    fancy_pre = fancy_pre + "\n" + resp['fancy_pre']
                    #print(f"got second pre: {resp['fancy_pre']}", "row_idx: ", row_idx)
                else:
                    fancy_pre = resp['fancy_pre']
            if resp['normal_pre']:
                if normal_pre is not None:
                    normal_pre = normal_pre + "\n" + resp['normal_pre']
                else:
                    normal_pre = resp['normal_pre']

        #copy over words if we have none
        words_tr = ' '.join(self.transcript_data[row_idx]['words_per_tr'])
        if not fancy_post and words_tr:
            fancy_post = words_tr


        #set post headers      
        if not self.exclude_post_dialogue_separator:
            if not fancy_post:
                fancy_post = "| No Dialogue |"
            else:
                #print(f"fancy_post: {fancy_post}")
                fancy_post = "| Dialogue |" + "\n" + fancy_post
            if not normal_post:
                normal_post = "| No Dialogue |"
            else:
                normal_post = "| Dialogue |" + "\n" + normal_post


        #now build rest of pre
        words_left = self.n_used_words
        if fancy_pre:
            words_left = self.n_used_words - len(fancy_pre.split())
        closest_dialogue = None

        
        #get scene of the dialogue
        if first_dialogue_in_row:
            closest_dialogue = first_dialogue_in_row
        else:
            #try to change row to previous row with dialogue
            closest_row_with_dialogue = self.get_closest_row_with_dialogue(row_idx)
            if closest_row_with_dialogue != -1:
                closest_dialogue = self.get_dialogues_for_row(closest_row_with_dialogue)[-1]
            else:
                closest_dialogue = None

        # if not fancy_post and not normal_post and closest_row_with_dialogue != -1: #row_idx - closest_row_with_dialogue > 2:
        #     print(f"closest_row_with_dialogue: {closest_row_with_dialogue}", "row_idx: ", row_idx)
        #     fancy_post = "(silence)"
        #     normal_post = "(silence)"

        # if closest_dialogue:
            #print(f"dataset closest_dialogue id: {closest_dialogue['id']}")

        if closest_dialogue:
            scene = get_scene_for_dialogue(closest_dialogue, self.scenes_and_dialogues)
            starting_diaglogue_id = closest_dialogue['id']
            scene_id = scene['id']
            used_summary = False
            while words_left > 0 and scene_id > 0 and not used_summary:   
                display_text, used_summary = get_scene_and_dialogues_display_text(self.scenes_and_dialogues, self.dialogue_list, scene_id, starting_diaglogue_id=starting_diaglogue_id, \
                    max_words=words_left, scene_summary=self.scene_summary, use_present_scene=self.use_present_scene)
                if display_text['fancy_scene_text']:
                    if fancy_pre:
                        fancy_pre = display_text['fancy_scene_text'] + "\n" + fancy_pre
                    else:
                        fancy_pre = display_text['fancy_scene_text']

                if display_text['normal_scene_text']:
                    if normal_pre:
                        normal_pre = display_text['normal_scene_text'] + "\n" + normal_pre
                    else:
                        normal_pre = display_text['normal_scene_text']
                words_left = words_left - len(display_text['fancy_scene_text'].split())
                starting_diaglogue_id = -1
                scene_id = scene_id - 1


        
        response = {
            "fancy_post": fancy_post,
            "normal_post": normal_post,
            "fancy_pre": fancy_pre,
            "normal_pre": normal_pre,
            "words_tr":words_tr,
            "word_length": words_left,
        }

        # print(f"response['fancy_post']: {response['fancy_post']}")
        # print(f"response['normal_post']: {response['normal_post']}")
        # print(f"response['fancy_pre']: {response['fancy_pre']}")
        # print(f"response['normal_pre']: {response['normal_pre']}")
        # print("!!!!!!!!!!ending for row: ", row_idx)
        return response