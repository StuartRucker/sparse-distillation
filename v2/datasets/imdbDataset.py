import random
import torch
from torch.utils.data import Dataset


class ImdbDataset(Dataset):
    def __init__(self, file_list, tokenizer, max_ngrams=1000, no_word_token=0):
        self.file_list = file_list
        self.tokenizer = tokenizer
        self.max_ngrams = max_ngrams
        self.no_word_token = no_word_token

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        
        #read self.file_list[idx] to a string
        review_contents = open(self.file_list[idx], "r").read()
        #tokenize the string
        feature_list = self.tokenizer.tokenize(review_contents)


       
        if len(feature_list) > self.max_ngrams:
            feature_list = random.sample(feature_list, self.max_ngrams)
        while len(feature_list) < self.max_ngrams:
            feature_list.append(self.no_word_token)
        label = [0,1] if 'pos' in self.file_list[idx] else [1,0]
        return torch.LongTensor(feature_list), torch.FloatTensor(label)

