import string
import pandas as pd
from glob import glob
from torch.utils.data import Dataset, DataLoader
import utils
from Scenes_and_dialogues import get_scene_dialogue, get_dialogue_list, match_dialogues_to_transcript_data, get_dialogue_display_text
from tabulate import tabulate

class SentenceDataset(Dataset):
    def __init__(self, sentences, mode="last_n_trs", last_n_trs=5, n_used_words=510):
        self.sentences = sentences
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

        if text== "": text= " "
        return text

class SentenceDataset_v2(Dataset):
    def __init__(self, transcript_data, scene_and_dialogues, tr_start, length):
        self.scenes_and_dialogues = scene_and_dialogues
        self.dialogue_list = get_dialogue_list(scene_and_dialogues)
        self.transcript_data = transcript_data
        self.dialogue_list, self.text_to_position_map, self.position_to_text_map = match_dialogues_to_transcript_data(self.transcript_data, self.dialogue_list)
        self.tr_start = tr_start
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError(f"Index {idx} is out of range for length {self.length}")
        data = {
            "pre_text": f"pre {idx}",
            "post_text": f"post {idx}",
        }
        return data
    
    def get_dialogues_for_row(self, row_idx):
        list_of_dialogues = []
        for dialogue in self.dialogue_list:
            if dialogue['matched_text_index_start'] != -1  and (dialogue['matched_row_index_start'] <= row_idx <= dialogue['matched_row_index_end']):
                list_of_dialogues.append(dialogue)
        #order dialogues by id
        list_of_dialogues.sort(key=lambda x: x['id'])
        return list_of_dialogues
    

    def _get_text_from_dialogue_for_row(self, dialogue, row_idx, allow_pre_text=False):
        #debug info
        # print(f"dialogue id: {dialogue['id']}")
        # print(f"dialogue normalized text: {dialogue['normalized_text']}")
        # print(f"dialogue length normalized text: {dialogue['length_normalized_text']}")
        # print(f"dialogue matched row index start: {dialogue['matched_row_index_start']}")
        # print(f"dialogue matched row index end: {dialogue['matched_row_index_end']}")
        # print(f"dialogue matched text index start: {dialogue['matched_text_index_start']}")
        # print(f"dialogue matched text index end: {dialogue['matched_text_index_end']}")
        # print(f"row position to text map: {self.position_to_text_map.get((row_idx, 0))}")
        response = " "
        row_word_length = len(self.transcript_data[row_idx]['words_per_tr'])
        #is dialogue starting in this row
        starting_in_this_row = dialogue['matched_row_index_start'] == row_idx
        #is dialogue ending in this row
        ending_in_this_row = dialogue['matched_row_index_end'] == row_idx

        if dialogue["length_normalized_text"] == 1 or (starting_in_this_row and ending_in_this_row):
            response1, response2 = get_dialogue_display_text(dialogue, withSpeaker=True)
        else:
            #we have to split this dialogue into multiple parts: middle and possibly pre and post
            #see how many words in the row overlap with the dialogue and which is the first and last word
            first_overlap_word_index = -1
            last_overlap_word_index = row_word_length - 1
            overlap_length = 0
            for word_idx in range(row_word_length):
                text_index = self.position_to_text_map.get((row_idx, word_idx))
                if dialogue['matched_text_index_start'] <= text_index <= dialogue['matched_text_index_end']:
                    overlap_length += 1
                    if first_overlap_word_index == -1:
                        first_overlap_word_index = text_index
                    last_overlap_word_index = text_index
            
            #starting index of dialogue for this row
            offset = max(0, first_overlap_word_index -dialogue['matched_text_index_start'] )
            start_index = round(offset*dialogue['run_rate'])
            dialogue_length_for_row = round(dialogue['run_rate'] * overlap_length)
            
        # print(f"first_overlap_word_index: {first_overlap_word_index}")
        # print(f"last_overlap_word_index: {last_overlap_word_index}")
            # print(f"overlap length: {overlap_length}")
            # print(f"offset: {offset}")
            # print(f"row_word_length: {row_word_length}")
            
            
            # print(f"run rate: {dialogue['run_rate']}")
        # print(f"dialogue length for row: {dialogue_length_for_row}")
        # print(f"start index: {start_index}")
            if dialogue_length_for_row + start_index > len(dialogue['normalized_text']):
                dialogue_length_for_row = dialogue_length_for_row - 1
            response1, response2 = get_dialogue_display_text(dialogue, withSpeaker=starting_in_this_row, start_index=start_index, length=dialogue_length_for_row)
            
            
        return response1, response2
            
            



    def get_post_text_for_tr(self, row_idx):
        response1, response2 = None, None
        word_length = len(self.transcript_data[row_idx]['words_per_tr'])
        if word_length == 0:
            response1 = "[No words spoken]"
            response2 = "[No words spoken]"
        else:
            list_of_dialogues = self.get_dialogues_for_row(row_idx)
            for dialogue in list_of_dialogues:
                resp1, resp2 = self._get_text_from_dialogue_for_row(dialogue, row_idx)
                if response1 is None:
                    response1 = resp1
                    response2 = resp2
                else:
                    response1 = response1 + "\n" + resp1
                    response2 = response2 + "\n" + resp2
        words_tr = ' '.join(self.transcript_data[row_idx]['words_per_tr'])
        
        return response1, response2, words_tr, word_length