
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
def get_wmt14_content(mini, language, mode):
    dataset_wmt14_en_train = load_from_disk(os.path.expanduser("~/hf_datasets/wmt14"))[mode]
    for i in range(len(dataset_wmt14_en_train)):
        yield dataset_wmt14_en_train[i]['translation'][language]
        if mini and i > 10:
            break
        if i > 5000000:
            break
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
    elif dataset_name == 'wmt14_en_train':
        return get_wmt14_content(mini, 'en', 'train')
    elif dataset_name == 'wmt14_de_train':
        return get_wmt14_content(mini, 'de', 'train')
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
    elif mode == 'ELECTRA':
        if dataset_name == 'wikibooks':
            return ElectraWikiDataset(tokenizer, mini=mini)
        else:
            raise ValueError(f'Invalid ELECTRA dataset name {dataset_name}')
    elif mode == "EMBED":
        if dataset_name == 'wikibooks':
            return EmbedWikiDataset(tokenizer, mini=mini)
        else:
            raise ValueError(f'Invalid EMBED dataset name {dataset_name}')
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


class ElectraWikiDataset(torch.utils.data.Dataset):
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

        before_index = max(0, mask_idx-5)
        after_index = min(before_index+7, len(tokenized_content))
        label = 0
        if random.random() < 0.5:

            # print("Deleting... ", tokenized_content[mask_idx])
            
            label = 1
            #replace masked worth with random token
            tokenized_content[mask_idx] = random.choice(list(self.bert_tokenizer.vocab.keys()))
            # print("Replacing with... ", tokenized_content[mask_idx])
            
        
        to_tokenize = tokenized_content[before_index:after_index]
        # if label == 1:
        #     print("Range: ", before_index, after_index, mask_idx)
        #     print(to_tokenize)
        
        #formerly call to get_features
        pad_dim,pad_value = 30, len(self.tokenizer.countvectorizer.vocabulary_)
        cx = self.tokenizer.transform_pretokenized(to_tokenize).tocoo()

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

        return torch.LongTensor(feature_list), torch.LongTensor([label])




def get_barlow_dataset(tokenizer_en, tokenizer_foreign, mini=False):
    return BarlowDataset(tokenizer_en, tokenizer_foreign, mini)

class BarlowDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer_en, tokenizer_foreign, mini=False):
        self.tokenizer_en = tokenizer_en
        self.tokenizer_foreign = tokenizer_foreign
        self.mini = mini

        self.dataset = load_from_disk(os.path.expanduser("~/hf_datasets/wmt14"))['train']
        self.length = len(self.dataset)
        print("Dataset length: ", self.length)
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # if self.mini:
        #     idx = idx % 50
        content_en = self.dataset[idx]['translation']['en']
        content_foreign = self.dataset[idx]['translation']['de']

        features_en = get_features(self.tokenizer_en, content_en, mask=False, pad_dim=1000, pad_value=len(self.tokenizer_en.countvectorizer.vocabulary_))
        features_foreign = get_features(self.tokenizer_foreign, content_foreign, mask=False, pad_dim=1000, pad_value=len(self.tokenizer_foreign.countvectorizer.vocabulary_))

        return torch.LongTensor(features_en), torch.LongTensor(features_foreign)

        


# dataset for which outputs the ngram ids, the positions of the ngrams, the length of the ngrams, and the position of the masked word
class EmbedWikiDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, mini=False):
        self.tokenizer = tokenizer
        tokenizer_path = os.path.join(os.path.dirname(__file__), "../data/bert_tokenizer")
        self.bert_tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

        self.dataset_books = load_from_disk(os.path.expanduser("~/hf_datasets/bookcorpus"))['train']
        self.dataset_wiki = load_from_disk(os.path.expanduser("~/hf_datasets/wikipedia"))['train']

        self.mini = mini

        terms = np.array(list(self.tokenizer.countvectorizer.vocabulary_.keys()))
        indices = np.array(list(self.tokenizer.countvectorizer.vocabulary_.values()))
        self.inverse_vocabulary = terms[np.argsort(indices)]

    def __len__(self):
        return len(self.dataset_books) + len(self.dataset_wiki)
    
    def __getitem__(self, idx):
        content = self.dataset_books[idx]['text'] if idx < len(self.dataset_books) else self.dataset_wiki[idx - len(self.dataset_books)]['text']
        
        tokenized_content = self.bert_tokenizer.tokenize(content)[:512]
        #select a random token to mask
        mask_idx = random.randint(0, len(tokenized_content) - 1)
        masked_word = tokenized_content[mask_idx]
        masked_word_idx = self.bert_tokenizer.convert_tokens_to_ids(masked_word)
        tokenized_content[mask_idx] = "[FAKEMASK]"

        start_index = max(0, mask_idx-10)
        end_index = min(start_index+20, len(tokenized_content))
        keep_tokens = tokenized_content[start_index:end_index]

        #disable mask mode
        self.tokenizer.transform(["Turning on mask mode"], mask=True)
        features = get_features(self.tokenizer, keep_tokens, mask=True, pad_dim=50, pad_value=len(self.tokenizer.countvectorizer.vocabulary_))
        
        ## Get the string associated with each feature index using self.inverse_vocabulary
        real_features = [f for f in features if f < len(self.inverse_vocabulary)]
        feature_strings = [self.inverse_vocabulary[feature_idx] for feature_idx in real_features]

        # get position of each feature_string in the original string
        stripped_content = " ".join(keep_tokens)
        index_to_position = []
        cnt = 0
        for tkn in keep_tokens:
            for k in range(len(tkn)+1):
                index_to_position += [cnt]
            cnt += 1
        
        
        # print(stripped_content)
        # print(feature_strings)
        positions = [index_to_position[stripped_content.find(feature_string)] for feature_string in feature_strings]
        lengths = [1+num_spaces(feature_string) for feature_string in feature_strings]

        # combine into one tensor 
        # (real_features, positions, lengths, [mask_idx-start_index for i in range(len(real_features))])
        x = torch.stack([torch.LongTensor(real_features), torch.LongTensor(positions), torch.LongTensor(lengths), torch.LongTensor([mask_idx-start_index for i in range(len(real_features))])], dim=0)

        # pad/truncate x to be length (4, 50)
        if x.shape[1] > 50:
            x = x[:, :50]
        if x.shape[1] < 50:
            x = torch.cat([x, torch.zeros((4, 50-x.shape[1]), dtype=torch.long)], dim=1)
        
        y = torch.LongTensor([masked_word_idx])
        return x, y

def num_spaces(s):
    return sum(1 for c in s if c == ' ')