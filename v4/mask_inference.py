import sys
import os
from dan import DAN
import torch
from tokenizer import Tokenizer
#import bert tokenizer
from transformers import BertTokenizer
import random
import numpy as np


#get the file path of first arguement
# model = DAN(embed_dimension=1000, intermediate_dimension=1000, num_embeddings=1000000, num_classes=2,)
# file_path = sys.argv[1]
file_path =  '/Users/stuartrucker/nlp/yoon/v4/cached/models/pretrain:[Corpus:IMDB_train Dtrain:IMDB_train pretrain:True kd:False fd:True mini:True]/49.model'

tokenizer = Tokenizer(None, "IMDB_train", max_features=1000000, mini=True)
tokenizer.transform(["[MASK]"], mask=True)

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

loaded = torch.load(file_path)

embed_dimension = loaded['model']['embed.weight'].shape[1]
num_embeddings = loaded['model']['embed.weight'].shape[0]
intermediate_dimension = loaded['model']['fc1.weight'].shape[1]
num_classes = loaded['model']['fc2.weight'].shape[0]
model = DAN(embed_dimension=embed_dimension, intermediate_dimension=intermediate_dimension, num_embeddings=num_embeddings, num_classes=num_classes)

model.load_state_dict(loaded['model'])


index_to_ngram = [''] * len(tokenizer)
for word, index in tokenizer.countvectorizer.vocabulary_.items():
    index_to_ngram[index] = word


def process(content):

    tokenized_content = bert_tokenizer.tokenize(content)[:512]
            #select a random token to mask


    cx =tokenizer.transform([tokenized_content], mask=True).tocoo()

    feature_list = []
    for i,j,v in zip(cx.row, cx.col, cx.data):
        feature_list += [j for _ in range(v)]

    ngrams = []
    for f in feature_list:
        ngrams.append(index_to_ngram[f])
    #if feature list is longer than 2000, randomly sample 2000 elements frmo it
    if len(feature_list) > 2000:
        feature_list = random.sample(feature_list, 2000)
    feature_list = np.array(feature_list)
    #if feature list is shorter than 2000, pad it with a value 1 greater than the size of the vocabulary
    if len(feature_list) < 2000:
        feature_list = np.pad(feature_list, (0, 2000 - len(feature_list)), 'constant', constant_values= len(tokenizer.countvectorizer.vocabulary_))


    feature_list = torch.LongTensor([feature_list])

    output = model(feature_list)

    #get the highest n indices
    top_n = output.topk(5)[1].detach().numpy()[0]

    #convert the indices to words using bert_tokenizer
    print(content, "\n\n")
    print(ngrams, '\n\n')

    for i in top_n:
        print("", bert_tokenizer.convert_ids_to_tokens([i])[0], ":", float(output[0][i].item()))


process("My name is Stuart [MASK] I love this movie")