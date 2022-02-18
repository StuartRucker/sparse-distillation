import sys
import os
import unittest
import torch
#append parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from data import get_content, get_ft_dataset
from tokenizer import Tokenizer

#Create suite of unit tests for Tokenizer
class Test(unittest.TestCase):
    
    def test_tokenizer(self):
        #test that tokenizer can be created
        
        tokenizer = Tokenizer("Amazon", "IMDB_train", max_features=100, mini=True)
        assert len(tokenizer.countvectorizer.get_feature_names_out()) > 20
    
    def test_dataset(self):
        #test that tokenizer can be created
        tokenizer = Tokenizer(None, "IMDB_train", max_features=100, mini=True)
        embeddings = torch.nn.Embedding(len(tokenizer.countvectorizer.get_feature_names_out()), 10)

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


        

