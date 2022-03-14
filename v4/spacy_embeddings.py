"""Preset the embeddings using spacy word2vec"""

import spacy
import torch
import pickle
import os
import numpy as np

"""Generates a pytorch weights object where embeddings are taken from spacy"""
def get_spacy_embeddings(tokenizer, embedding_size):
    # join the path of this file with ../../data/spacy.model
    spacy_model_path = os.path.join(os.path.dirname(__file__), "../data/spacy.pickle")
    print("Loading spacy model from {}".format(spacy_model_path))
    nlp = pickle.load(open(spacy_model_path, "rb"))
    embeddings = torch.zeros(len(tokenizer)+1, embedding_size)
    for ngram, i in tokenizer.countvectorizer.vocabulary_.items():
        ngram = ngram.replace(" ##", "")
        ngram = ngram.replace("##", "")
        doc = nlp(ngram)
        found_cnt = 0
        for token in doc:
            
            if token.has_vector:
                base_embedding = torch.from_numpy(np.array(token.vector).copy())
                final_embedding = base_embedding.clone()
                while final_embedding.shape[0] < embedding_size:
                    final_embedding = torch.cat((final_embedding, base_embedding))
                if final_embedding.shape[0] > embedding_size:
                    final_embedding = final_embedding[:embedding_size]
                embeddings[i] += final_embedding 
                found_cnt += 1
        if found_cnt > 0:
            embeddings[i] /= float(found_cnt)
        # print(i, ngram, embeddings[i][:10])
            #tile embedding so it is the same size as embedding_size
    return embeddings

