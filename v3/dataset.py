"""Loads all information"""

import random
import re
import glob
import json
import time

from fastCountVectorizer import FastCountVectorizer

from transformers import DistilBertTokenizer, DistilBertForMaskedLM
import torch
from torch.utils.data import Dataset, IterableDataset








class ReviewsRaw():
    def __init__(self, filenames, ngram_tokenizer, tokenizer, max_ngrams=1000, no_word_token=0, lines_per_file = 10000):
        self.ngram_tokenizer = ngram_tokenizer
        self.filenames = filenames
        self.no_word_token = no_word_token
        self.max_ngrams = max_ngrams
        self.lines_per_file = lines_per_file


        #gets review at index
        self.tokenizer = tokenizer
        
        

        
        
    def __len__(self):
        return len(self.filenames) * self.lines_per_file
    
    def __getitem__(self, idx):
        file_id = idx // self.lines_per_file
        

        with open(self.filenames[file_id]) as f:
            lines = f.readlines()
            line_id = idx % len(lines)

            review_obj = json.loads(lines[line_id])
            
            text = review_obj['text']

            return self.transform(text)
    
    def get_allwords(self, text):
        all_words = re.findall( r'(?u)\b\w\w+\b', text.lower())
        return all_words[:500]

    # replace a random word with [MASK]
    def mask_word(self, words):
        replace_index = random.randint(0, len(words)-1)
        former_word = words[replace_index]
        words[replace_index] = "[MASK]"
        return former_word

    def extract_ngrams(self, all_words, max_range=4):
        ngrams = []
        for i in range(len(all_words)):
            for j in range(i+1, 1+min(i+max_range, len(all_words))):
                ngrams.append(' '.join(all_words[i:j]))
        return ngrams

    def pad(self, ngram_ids):
        #if ngram_ids is too long, randomly sample self.max_ngrams. If too short, pad with self.no_word_token
        if len(ngram_ids) > self.max_ngrams:
            ngram_ids = random.sample(ngram_ids, self.max_ngrams)
        else:
            ngram_ids += [self.no_word_token] * (self.max_ngrams - len(ngram_ids))
        return ngram_ids

    #transforms text to correct format
    def transform(self, text):
        all_words = self.get_allwords(text)
        masked_word = self.mask_word(all_words)
        masked_text = " ".join(all_words)

        bert_tokens_str = self.tokenizer.tokenize(masked_text)
        
        masked_index = bert_tokens_str.index("[MASK]")+1
        
        masked_ngrams = self.extract_ngrams(all_words)
        ngram_ids = self.pad(self.ngram_tokenizer.tokenize(masked_ngrams))
        return masked_text, masked_index, ngram_ids

class ReviewBatcher():
    def __init__(self, raw_dataset, tokenizer, batch_size = 2):
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.softmax = torch.nn.Softmax(dim=1)

        self.model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")
        self.model.eval()
        
    def classifier(self, text, mask_indices):
        with torch.no_grad():
            tokenized_text = self.tokenizer(text, max_length=512, truncation=True, padding=True, return_tensors="pt").to(self.model.device)
            model_output = self.model(**tokenized_text)

            #Get the logits at the masked index for batch_index
            logits = self.softmax(model_output.logits)
            used_logits = []
            for i, mask_index in enumerate(mask_indices):
                used_logits.append(logits[i, mask_index, :])
            return torch.vstack(used_logits)

    def get_batch(self):
        indices = random_indices = [random.randint(0, len(self.raw_dataset)-1) for i in range(self.batch_size)]

        
        batch_info = [self.raw_dataset[i] for i in indices]
        #unzip this list of tuples
        batch_text, mask_indices, batched_ngrams = zip(*batch_info)

        print(batch_text)
        return  torch.IntTensor(batched_ngrams), self.classifier(batch_text, mask_indices)





if __name__ == '__main__':
    #using glob, get all files in ../data/sentiment_reviews/
    filenames = glob.glob("../data/sentiment_reviews/*")
    ngram_tokenizer = FastCountVectorizer("../data/fast_countvectorizer/top_ngrams_masked.txt")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    rawdataset = ReviewsRaw(filenames, ngram_tokenizer, tokenizer)
    reviewbatcher = ReviewBatcher(rawdataset, tokenizer)

    batch = reviewbatcher.get_batch()

