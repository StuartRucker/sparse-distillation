
import json
import os
import glob
import random
import torch
import numpy as np
#import BertTokenizer
from transformers import BertTokenizer 
from datasets import load_from_disk

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


def get_wikibooks_content(mini=False, limit_wiki=100000, limit_books=500000):
    dataset_books = load_from_disk(os.path.expanduser("~/hf_datasets/bookcorpus"))['train']
    dataset_wiki = load_from_disk(os.path.expanduser("~/hf_datasets/wikipedia"))['train']
    
    for i in range(len(dataset_books)):
        yield dataset_books[i]['text']
        if mini and i > 10:
            break
        if i > limit_books:
            break
    for i in range(len(dataset_wiki)):
        yield dataset_wiki[i]['text']
        if mini and i > 10:
            break
        if i > limit_wiki:
            break
    # for i in range(len(dataset_imdb)):
    #     yield dataset_imdb[i]['text'].strip().replace('<br />', '\n')
    #     if mini and i > 10:
    #         break

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
    elif dataset_name == 'wikibooks':
        return get_wikibooks_content(mini)
    else:
        raise ValueError('Invalid dataset name')


def get_pretrain_dataset(dataset_name, tokenizer, mini=False, mode='DAN'):
    if mode == 'DAN':
        if dataset_name.startswith('IMDB'):
            path = os.path.join(os.path.dirname(__file__), "../data/imdb/train/*/*.txt" if 'train' in dataset_name else "../data/imdb/test/*/*.txt")
            files = glob.glob(path)
            if mini:
                random.shuffle(files)
                files = files[:100]

            return MaskImdbDataset(files, tokenizer)
        elif dataset_name == 'wikibooks':
            return MaskWikiDataset(tokenizer, mini=mini)
        else:
            raise ValueError(f'Invalid dataset name {dataset_name}')
    elif mode == 'CBOW':
        if dataset_name == 'wikibooks':
            return CBOWWikiDataset(tokenizer, mini=mini)
        else:
            raise ValueError(f'Invalid CBOW dataset name {dataset_name}')
    else:
        raise ValueError(f'Invalid pretrain mode {mode}')

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


def get_features(tokenizer, phase, pad_value, pad_dim=2000, mask=False):
        cx = tokenizer.transform([phase], mask=mask).tocoo()
        feature_list = []
        for i,j,v in zip(cx.row, cx.col, cx.data):
            feature_list += [j for _ in range(v)]
        
        #if feature list is longer than 2000, randomly sample 2000 elements frmo it
        if len(feature_list) > pad_dim:
            feature_list = random.sample(feature_list, pad_dim)
        feature_list = np.array(feature_list)
        #if feature list is shorter than 2000, pad it with a value 1 greater than the size of the vocabulary
        if len(feature_list) < pad_dim:
            feature_list = np.pad(feature_list, (0, pad_dim - len(feature_list)), 'constant', constant_values= pad_value)
        return feature_list

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
        tokenizer_path = os.path.join(os.path.dirname(__file__), "../data/bert_tokenizer")
        self.bert_tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

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


# TODO, use inheritance to combine This and MaskImdbDataset
class MaskWikiDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, mini=False):
        self.tokenizer = tokenizer
        tokenizer_path = os.path.join(os.path.dirname(__file__), "../data/bert_tokenizer")
        self.bert_tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

        self.dataset_books = load_from_disk(os.path.expanduser("~/hf_datasets/bookcorpus"))['train']
        self.dataset_wiki = load_from_disk(os.path.expanduser("~/hf_datasets/wikipedia"))['train']
        
        self.mini = mini

        self.length = len(self.dataset_books) + len(self.dataset_wiki)
        print("Dataset length: ", self.length)
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        
        if self.mini:
            idx = idx % 50
        if idx < len(self.dataset_books):
            content = self.dataset_books[idx]['text']
        else:
            content = self.dataset_wiki[idx - len(self.dataset_books)]['text']

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

class CBOWWikiDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, mini=False):
        self.tokenizer = tokenizer
        tokenizer_path = os.path.join(os.path.dirname(__file__), "../data/bert_tokenizer")
        self.bert_tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

        self.dataset_books = load_from_disk(os.path.expanduser("~/hf_datasets/bookcorpus"))['train']
        self.dataset_wiki = load_from_disk(os.path.expanduser("~/hf_datasets/wikipedia"))['train']
        
        self.mini = mini

        self.length = len(self.dataset_books) + len(self.dataset_wiki)
        print("Dataset length: ", self.length)
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        
        if self.mini:
            idx = idx % 50
        if idx < len(self.dataset_books):
            content = self.dataset_books[idx]['text']
        else:
            content = self.dataset_wiki[idx - len(self.dataset_books)]['text']

        tokenized_content = self.bert_tokenizer.tokenize(content)[:512]
        #select a random token to mask
        mask_idx = random.randint(0, len(tokenized_content) - 1)
        masked_word = tokenized_content[mask_idx]
        masked_word_idx = self.bert_tokenizer.convert_tokens_to_ids(masked_word)

        before_tokens = tokenized_content[max(0, mask_idx-10):mask_idx]
        after_tokens = tokenized_content[mask_idx+1:min(mask_idx+11, len(tokenized_content))]

        #use convert_tokens_to_string
        before_phase = self.bert_tokenizer.convert_tokens_to_string(before_tokens)
        after_phrase = self.bert_tokenizer.convert_tokens_to_string(after_tokens)

        #disable mask mode
        self.tokenizer.transform(["Turning off mask mode"], mask=False)
        features_before = get_features(self.tokenizer, before_phase, mask=False, pad_dim=30, pad_value=len(self.tokenizer.countvectorizer.vocabulary_))
        features_after = get_features(self.tokenizer, after_phrase, mask=False, pad_dim=30, pad_value=len(self.tokenizer.countvectorizer.vocabulary_))

         # get the bert index of the masked_word
        return torch.stack([torch.LongTensor(features_before),torch.LongTensor(features_after)]), torch.LongTensor([masked_word_idx])

        
