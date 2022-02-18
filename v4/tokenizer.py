#import hugging face BERT tokenizer
from transformers import BertTokenizer
import os
from sklearn.feature_extraction.text import CountVectorizer
from data import get_content
import pickle

#create a class Tokenizer
class Tokenizer:
    def __init__(self, corpus, d, max_features=1000000, mini=False):
        self.corpus = corpus
        self.d = d
        self.name = f"{self.corpus}&{self.d}" + (".mini" if mini else "") + ".tokenizer"
        self.max_features = max_features
        self.mini = mini

        
        self.path = os.path.join(os.path.dirname(__file__), "cached/tokenizers", self.name)
        
        #if this file exists, read it
        if os.path.exists(self.path):
            print("Loading tokenizer from cache...")
            with open(self.path, "rb") as f:
                self.countvectorizer = pickle.load(f)
        else:
            print("Computing tokenizer from scratch...")
            self.create_countvectorizer()
    
    def create_countvectorizer(self):
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.countvectorizer = CountVectorizer(ngram_range=(1, 4), token_pattern=None,
                 max_features=self.max_features, input='content', tokenizer=bert_tokenizer.tokenize, )
        
        content = get_content(self.corpus, self.mini) + get_content(self.d, self.mini)
        self.countvectorizer.fit(content)
        
        #pickle the countvectorizer
        with open(self.path, "wb") as f:
            pickle.dump(self.countvectorizer, f)
    def transform(self, text):
        return self.countvectorizer.transform(text)

        
        
    # tokenizer.encode_plus(text, add_special_tokens = True,    truncation = True, padding = "max_length", return_attention_mask = True, return_tensors = "pt")

        