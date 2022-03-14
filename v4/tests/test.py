import sys
import os
import unittest
import torch
from transformers import BertTokenizer
from spacy_embeddings import get_spacy_embeddings
#append parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from data import get_content, get_ft_dataset, get_pretrain_dataset
from tokenizer import Tokenizer

#Create suite of unit tests for Tokenizer
class Test(unittest.TestCase):
    
    def test_tokenizer(self):
        #test that tokenizer can be created
        
        tokenizer = Tokenizer("Amazon", "IMDB_train", max_features=100, mini=True)
        assert len(tokenizer.countvectorizer.get_feature_names()) > 20
    
    def test_mask_tokenizer(self):
        #test that tokenizer can be created
        
        tokenizer = Tokenizer(None, "IMDB_train", max_features=100, mini=True)
        original_vocab_length = len(tokenizer.countvectorizer.vocabulary_)

        tokenizer_path = os.path.join(os.path.dirname(__file__), "../../data/bert_tokenizer")
        bert_tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        tokenized_text = bert_tokenizer.tokenize("I love this movie")
        tokenized_text[2] = "[MASK]"
        transformed = tokenizer.transform([tokenized_text], mask=True)
        final_vocab_length = len(tokenizer.countvectorizer.vocabulary_)
        assert final_vocab_length > original_vocab_length, "Vocabulary should have been expanded"

        words = []
        for i in range(transformed.shape[0]):
            for j in range(transformed.shape[1]):
                if transformed[i,j] > 0:
                    words.append(tokenizer.countvectorizer.get_feature_names_out()[j])

        assert any(['[MASK]' in word for word in words]), "Tokenizer incorporated mask token"
    
    def test_pretrain_dataset(self):
        tokenizer = Tokenizer(None, "IMDB_train", max_features=100, mini=True)
        dataset = get_pretrain_dataset("IMDB_train", tokenizer, mini=True)
        assert len(dataset) > 5
        for x, y in dataset:
            print(y)

    def test_dataset(self):
        #test that tokenizer can be created
        tokenizer = Tokenizer(None, "IMDB_train", max_features=100, mini=True)
        embeddings = torch.nn.Embedding(len(tokenizer.countvectorizer.get_feature_names()), 10)

        dataset = get_ft_dataset("IMDB_train", tokenizer, embeddings, mini=True)

        assert len(dataset) > 5
        
        for x, y in dataset:
            assert x.shape == torch.Size([10])
            assert y.shape == torch.Size([2])

        
    def test_dataset(self):
        content = get_content("Amazon", mini=True)
        assert len(content) > 4

        content = get_content("IMDB_train", mini=True)
        assert len(content) > 50
        content = get_content("IMDB_test", mini=True)
        assert len(content) > 50

    def test_wikibooks(self):
        content = get_content("wikibooks", mini=True)
        for c in content:
            print(c)
        
    def test_cbow_dataset(self):
        tokenizer = Tokenizer(None, "wikibooks", max_features=100, mini=True)
        content = get_pretrain_dataset("wikibooks",tokenizer, mini=True, mode='CBOW')
        
        tokenizer_path = os.path.join(os.path.dirname(__file__), "../../data/bert_tokenizer")
        bert_tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

        index_to_ngram = {}
        for ngram, index in tokenizer.countvectorizer.vocabulary_.items():
            index_to_ngram[index] = ngram
        index_to_ngram[100]='PAD'
        print(index_to_ngram)
        for (before_indices, after_indices), label in content:
            print("before:")
            for b in before_indices:
                print(index_to_ngram[int(b)])
            print("after:")
            for b in after_indices:
                print(index_to_ngram[int(b)])
            print("label:")
            print(bert_tokenizer.convert_ids_to_tokens([int(label[0])]))
            
    def test_spacy(self):
        tokenizer = Tokenizer(None, "wikibooks", max_features=100, mini=True)

        embeddings = get_spacy_embeddings(tokenizer, 1000)
        
        #assert that the embeddings are not all zeros
        assert torch.std(embeddings) > .01
        


        

