import string
import pandas as pd
from glob import glob
from torch.utils.data import Dataset, DataLoader
import utils
import torch
import gzip
import pickle
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