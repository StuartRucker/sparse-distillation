from torch.utils.data import IterableDataset
import random
import os
import torch
import json


class RandomReviews(IterableDataset):
    def __init__(self, filenames, tokenizer, max_ngrams=500, no_word_token=0):
        self.tokenizer = tokenizer
        self.filenames = filenames
        self.no_word_token = no_word_token
        self.max_ngrams = max_ngrams
        self.stream = os.popen("cat " + " ".join(self.filenames) + " | shuf")
    
    def __len__(self):
        return int(os.popen("cat " + " ".join(self.filenames) + " | wc -l").read())


    def __iter__(self):
        line = self.stream.readline()
        while line:
            review_obj = json.loads(line)
            feature_list = self.tokenizer.tokenize(review_obj['text'])

            if len(feature_list) > self.max_ngrams:
                feature_list = random.sample(feature_list, self.max_ngrams)
            while len(feature_list) < self.max_ngrams:
                feature_list.append(self.no_word_token)

            yield (torch.LongTensor(feature_list), torch.FloatTensor([review_obj['val']]))
            line = self.stream.readline()
