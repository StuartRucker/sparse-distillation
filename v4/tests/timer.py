import sys
import os
import unittest
import torch
import time
#append parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import get_content, get_ft_dataset
from tokenizer import Tokenizer

def time_tokenizer():
    tokenizer = Tokenizer(None, "IMDB_train", max_features=1000000, mini=True)
    content = get_content("IMDB_train", mini=True)
    start = time.time()
    for blurb in content:
        tokenizer.tokenize(blurb)
    print("Tokenizer took {} seconds to process {} blurbs".format(time.time() - start, len(content)))
time_tokenizer()