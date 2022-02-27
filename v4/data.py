
import json
import os
import glob
import random
import torch
import numpy as np
#import BertTokenizer
from transformers import BertTokenizer 


def get_amazon_content(path):
    #read the file from path
    content = []
    with open(path, 'r') as f:
        #for each line in f, read into json object
        for line in f:
            json_object = json.loads(line)
            content.append(json_object['reviewText'])
    return content

def get_imdb_content(mini, train=True):
    path = os.path.join(os.path.dirname(__file__), "../data/imdb/train/*/*.txt" if train else "../data/imdb/test/*/*.txt")
    files = glob.glob(path)
    if mini:
        files = files[:55]

    content = []
    for file in files:
        with open(file, 'r') as f:
            content.append(f.read().strip().replace('<br />', '\n'))
    return content

    #returns a list of strings for each given dataset name


#returns a list of strings for each given dataset name
def get_content(dataset_name, mini=False):
    if not dataset_name:
        return []
    if dataset_name == 'Amazon':
        #join the path of this file with tests/testdata/amazon.jsonl
        return get_amazon_content(os.path.join(os.path.dirname(__file__), "tests/testdata/amazon.jsonl")) if mini else None
    elif dataset_name == 'IMDB_train':
        return get_imdb_content(mini, train=True)
    elif dataset_name == 'IMDB_test':
        return get_imdb_content(mini, train=False)
    else:
        raise ValueError('Invalid dataset name')


def get_pretrain_dataset(dataset_name, tokenizer, mini=False):
    if dataset_name.startswith('IMDB'):
        path = os.path.join(os.path.dirname(__file__), "../data/imdb/train/*/*.txt" if 'train' in dataset_name else "../data/imdb/test/*/*.txt")
        files = glob.glob(path)
        if mini:
            random.shuffle(files)
            files = files[:100]

        return MaskImdbDataset(files, tokenizer)
    else:
        raise ValueError(f'Invalid dataset name {dataset_name}')

def get_ft_dataset(dataset_name, tokenizer, mini=False):
    if dataset_name.startswith('IMDB'):
        path = os.path.join(os.path.dirname(__file__), "../data/imdb/train/*/*.txt" if 'train' in dataset_name else "../data/imdb/test/*/*.txt")
        files = glob.glob(path)
        if mini:
            random.shuffle(files)
            files = files[:100]

        return ImdbDataset(files, tokenizer.countvectorizer)
    else:
        raise ValueError(f'Invalid dataset name {dataset_name}')


class ImdbDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, trained_vectorizer):
        self.file_list = file_list
        self.trained_vectorizer = trained_vectorizer

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        with open(self.file_list[idx], 'r') as f:
            content = f.read().strip().replace('<br />', '\n')
        cx = self.trained_vectorizer.transform([content]).tocoo()
        
        feature_list = []
        for i,j,v in zip(cx.row, cx.col, cx.data):
            feature_list += [j for _ in range(v)]



        #if feature list is longer than 2000, randomly sample 2000 elements frmo it
        if len(feature_list) > 2000:
            feature_list = random.sample(feature_list, 2000)
        feature_list = np.array(feature_list)
        #if feature list is shorter than 2000, pad it with a value 1 greater than the size of the vocabulary
        if len(feature_list) < 2000:
            feature_list = np.pad(feature_list, (0, 2000 - len(feature_list)), 'constant', constant_values= len(self.trained_vectorizer.vocabulary_))

        label = [1] if 'pos' in self.file_list[idx] else [0]


        return torch.LongTensor(feature_list), torch.LongTensor(label)


class MaskImdbDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, tokenizer):
        self.file_list = file_list
        self.tokenizer = tokenizer
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        with open(self.file_list[idx], 'r') as f:
            content = f.read().strip().replace('<br />', '\n')
        tokenized_content = self.bert_tokenizer.tokenize(content)[:512]
        #select a random token to mask
        mask_idx = random.randint(0, len(tokenized_content) - 1)
        masked_word = tokenized_content[mask_idx]
        tokenized_content[mask_idx] = '[MASK]'

        # get the bert index of the masked_word
        masked_word_idx = self.bert_tokenizer.convert_tokens_to_ids(masked_word)


        cx = self.tokenizer.transform([tokenized_content], mask=True).tocoo()
        
        feature_list = []
        for i,j,v in zip(cx.row, cx.col, cx.data):
            feature_list += [j for _ in range(v)]
        

        #if feature list is longer than 2000, randomly sample 2000 elements frmo it
        if len(feature_list) > 2000:
            feature_list = random.sample(feature_list, 2000)
        feature_list = np.array(feature_list)
        #if feature list is shorter than 2000, pad it with a value 1 greater than the size of the vocabulary
        if len(feature_list) < 2000:
            feature_list = np.pad(feature_list, (0, 2000 - len(feature_list)), 'constant', constant_values= len(self.tokenizer.countvectorizer.vocabulary_))
        return torch.LongTensor(feature_list), torch.LongTensor([masked_word_idx])
