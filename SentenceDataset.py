import string
import pandas as pd
from glob import glob
from torch.utils.data import Dataset, DataLoader
import utils
import torch
import gzip
import pickle
from Scenes_and_dialogues import get_scene_dialogue, get_dialogue_list, match_dialogues_to_transcript_data

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