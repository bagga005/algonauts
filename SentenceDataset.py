import string
import pandas as pd
from glob import glob
from torch.utils.data import Dataset, DataLoader
import utils
from Scenes_and_dialogues import get_scene_dialogue, get_dialogue_list, match_dialogues_to_transcript_data, get_dialogue_display_text, get_scene_for_dialogue, get_scene_display_text, get_closest_dialogue_for_row, get_scene_and_dialogues_display_text
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
    def __init__(self, transcript_data, scene_and_dialogues, tr_start, length, n_used_words=1000):
        self.scenes_and_dialogues = scene_and_dialogues
        self.dialogue_list = get_dialogue_list(scene_and_dialogues)
        self.transcript_data = transcript_data
        self.dialogue_list, self.text_to_position_map, self.position_to_text_map = match_dialogues_to_transcript_data(self.transcript_data, self.dialogue_list)
        self.tr_start = tr_start
        self.length = length
        self.n_used_words = n_used_words

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
        starting_in_this_row = dialogue['matched_row_index_start'] == row_idx
        #is dialogue ending in this row
        ending_in_this_row = dialogue['matched_row_index_end'] == row_idx

        if dialogue["length_normalized_text"] == 1 or (starting_in_this_row and ending_in_this_row):
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
                if dialogue['matched_text_index_start'] <= text_index <= dialogue['matched_text_index_end']:
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

            if start_index > 0:
                add_prefix_continuation_for_post = True
            if start_index + dialogue_length_for_row < len(dialogue['normalized_text']):
                add_suffix_continuation_for_post = True

            #prepare response object
            #print(f" calling for post get_dialogue_display_text start_index: {start_index}, length: {dialogue_length_for_row}")
            response_post = get_dialogue_display_text(dialogue, withSpeaker=starting_in_this_row or forcePostSpeaker, start_index=start_index, length=dialogue_length_for_row,
                                                      add_prefix_continuation=add_prefix_continuation_for_post, add_suffix_continuation=add_suffix_continuation_for_post)

            #get pre text if needed
            if start_index > 0:
                #print(f" calling for pre get_dialogue_display_text start_index: {0}, length: {start_index }")
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
        def get_dialogue_by_id(dialogue_id):
            for dialogue in self.dialogue_list:
                if dialogue['id'] == dialogue_id:
                    return dialogue
            return None
        
        def get_scene_by_id(scene_id):
            for scene in self.scenes_and_dialogues['scenes']:
                if scene['id'] == scene_id:
                    return scene
            return None

        #print("!!!!!!!!!!starting for row: ", row_idx)
        fancy_post, normal_post, fancy_pre, normal_pre = None, None, None, None
        word_length = len(self.transcript_data[row_idx]['words_per_tr'])
        first_dialogue_in_row = None
        #setup post main text
        list_of_dialogues = self.get_dialogues_for_row(row_idx)
        for dialogue in list_of_dialogues:
            if first_dialogue_in_row is None:
                first_dialogue_in_row = dialogue
            forcePostSpeaker = normal_post is not None
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
        if not fancy_post:
            fancy_post = "|No Dialogue|"
        else:
            #print(f"fancy_post: {fancy_post}")
            fancy_post = "| Dialogue |" + "\n" + fancy_post
        if not normal_post:
            normal_post = "|No Dialogue|"
        else:
            normal_post = "| Dialogue |" + "\n" + normal_post
        
        



        #now build rest of pre
        words_left = self.n_used_words
        if fancy_pre:
            words_left = self.n_used_words - len(fancy_pre.split())
        closest_dialogue = None

        # if first_dialogue_in_row:   
            #print(f"dataset frist dialogue id: {first_dialogue_in_row['id']}")

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

        # if closest_dialogue:
            #print(f"dataset closest_dialogue id: {closest_dialogue['id']}")

        if closest_dialogue:
            scene = get_scene_for_dialogue(closest_dialogue, self.scenes_and_dialogues)
            starting_diaglogue_id = closest_dialogue['id']
            scene_id = scene['id']
            while words_left > 0 and scene_id > 0:   
                display_text = get_scene_and_dialogues_display_text(self.scenes_and_dialogues, self.dialogue_list, scene_id, starting_diaglogue_id=starting_diaglogue_id, max_words=words_left)
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