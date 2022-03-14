"""Preset the embeddings using spacy word2vec"""

import spacy
import torch

"""Generates a pytorch weights object where embeddings are taken from spacy"""
def get_spacy_embeddings(tokenizer, embedding_size):
    nlp = spacy.load('en_core_web_lg')
    embeddings = torch.zeros(len(tokenizer)+1, embedding_size)
    for ngram, i in tokenizer.countvectorizer.vocabulary_.items():
        ngram = ngram.replace(" ##", "")
        ngram = ngram.replace("##", "")
        doc = nlp(ngram)
        found_cnt = 0
        for token in doc:
            
            if token.has_vector:
                base_embedding = torch.from_numpy(token.vector)
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

