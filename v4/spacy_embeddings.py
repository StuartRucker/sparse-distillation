"""Preset the embeddings using spacy word2vec"""


import torch
import pickle
import os
import numpy as np
import tqdm

from gensim.models.word2vec import Word2Vec
import gensim.downloader as api


"""Generates a pytorch weights object where embeddings are taken from spacy"""
def get_spacy_embeddings(tokenizer, embedding_size):
   
    corpus = api.load('text8')
    

    # create word2vec model
    model = Word2Vec(corpus, vector_size=embedding_size)
    

    embeddings = torch.zeros(len(tokenizer)+1, embedding_size)
    for ngram, i in tqdm.tqdm(tokenizer.countvectorizer.vocabulary_.items()):
        ngram = ngram.replace(" ##", "")
        ngram = ngram.replace("##", "")
        
        found_cnt = 0
        for token in ngram.split(" "):
            
            if token in model.wv:
                embeddings[i] += model.wv[token] 
                found_cnt += 1
        if found_cnt > 0:
            embeddings[i] /= float(found_cnt)
      
        return embeddings

