#import hugging face BERT tokenizer
from transformers import BertTokenizer
import os
from sklearn.feature_extraction.text import CountVectorizer
from data import get_content
import pickle
from itertools import chain

#create a class Tokenizer
class Tokenizer:
    def __init__(self, corpus, d, max_features=1000000, mini=False):
        self.corpus = corpus
        self.d = d
        self.name = f"{self.corpus}&{self.d}" + (".mini" if mini else "") + ".tokenizer"
        self.max_features = max_features
        self.mini = mini

        self.mask_mode = False
        self.mask_vocabulary = {}

        self.path = os.path.join(os.path.dirname(__file__), "cached/tokenizers", self.name)
        
        #if this file exists, read it
        if os.path.exists(self.path):
            print("Loading tokenizer from cache...")
            with open(self.path, "rb") as f:
                self.countvectorizer = pickle.load(f)
        else:
            print("Computing tokenizer from scratch...")
            self.create_countvectorizer()
        self.normal_vocabulary = self.countvectorizer.vocabulary_

    def __len__(self):
        return len(self.countvectorizer.vocabulary_)

    def create_countvectorizer(self):
        #join path of this file to "../data/bert_tokenizer"
        tokenizer_path = os.path.join(os.path.dirname(__file__), "../data/bert_tokenizer")
        bert_tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

        # tokenize_wrapper = lambda x: bert_tokenizer.tokenize(x)[:512]
        self.countvectorizer = CountVectorizer(ngram_range=(1, 4), max_features=self.max_features,
                input='content', tokenizer=bert_tokenizer.tokenize, lowercase=False)
        

        content = chain(get_content(self.corpus, self.mini),get_content(self.d, self.mini))
        self.countvectorizer.fit(content)
        
        #pickle the countvectorizer
        with open(self.path, "wb") as f:
            pickle.dump(self.countvectorizer, f)
    def generate_mask_vocabulary(self):
        print("Generating mask vocabulary...")
        self.mask_vocabulary = {}
        next_index = len(self.countvectorizer.vocabulary_)

        for ngram, index in self.countvectorizer.vocabulary_.items():
            words = ngram.split()
            if len(words) < 4:
                for i in range(len(words)):
                    #replace ith word with <MASK>
                    new_ngram = " ".join(words[:i] + ["[MASK]"] + words[i+1:])
                    if new_ngram not in self.mask_vocabulary:
                        self.mask_vocabulary[new_ngram] = next_index
                        next_index += 1
            self.mask_vocabulary[ngram] = index
    def skip_func(self, x):
        return x
    def transform(self, text, mask=False):
        # if mask mode expects the text to be tokenized
        if mask and not self.mask_mode:
            if not self.mask_vocabulary:
                self.generate_mask_vocabulary()
            self.normal_vocabulary = self.countvectorizer.vocabulary_
            self.normal_tokenizer = self.countvectorizer.tokenizer
            self.countvectorizer.tokenizer = self.skip_func
            self.countvectorizer.vocabulary_ = self.mask_vocabulary
            
            self.mask_mode = True
        elif not mask and self.mask_mode:
            self.countvectorizer.vocabulary_ = self.normal_vocabulary
            self.countvectorizer.tokenizer = self.normal_tokenizer
            self.mask_mode = False

        return self.countvectorizer.transform(text)
    
    # returns the vocabulary size of the hugging face BertTokenizer
    def get_bert_vocabulary_size(self):
        tokenizer_path = os.path.join(os.path.dirname(__file__), "../data/bert_tokenizer")
        bert_tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        return len(bert_tokenizer.vocab)

        
    # tokenizer.encode_plus(text, add_special_tokens = True,    truncation = True, padding = "max_length", return_attention_mask = True, return_tensors = "pt")

        