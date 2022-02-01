from torch.utils.data import Dataset, IterableDataset
import random
import os
import torch
import json
import subprocess
import itertools


class SentimentReviews(Dataset):
    def __init__(self, filenames, tokenizer, max_ngrams=1000, no_word_token=0):
        self.tokenizer = tokenizer
        self.filenames = filenames
        self.no_word_token = no_word_token
        self.max_ngrams = max_ngrams
        self.lines = []
        for filename in filenames:
            with open(filename, 'r') as f:
                self.lines.extend(f.readlines())
   
    def __len__(self):
        return len(self.lines)
    

    def __getitem__(self, idx):
        review_obj = json.loads(self.lines[idx])
        
        feature_list = review_obj['tokens']

        if len(feature_list) > self.max_ngrams:
            feature_list = random.sample(feature_list, self.max_ngrams)
        while len(feature_list) < self.max_ngrams:
            feature_list.append(self.no_word_token)
        
        return (torch.LongTensor(feature_list), torch.FloatTensor([1-review_obj['val'], review_obj['val']]))
       
