
import json
import os
import glob
import random
import torch



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
            content.append(f.read())
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


def get_ft_dataset(dataset_name, tokenizer, torch_embeddings, mini=False):
    if dataset_name.startswith('IMDB'):
        path = os.path.join(os.path.dirname(__file__), "../data/imdb/train/*/*.txt" if 'train' in dataset_name else "../data/imdb/test/*/*.txt")
        files = glob.glob(path)
        if mini:
            random.shuffle(files)
            files = files[:100]

        return ImdbDataset(files, tokenizer.countvectorizer, torch_embeddings)
    else:
        raise ValueError(f'Invalid dataset name {dataset_name}')


class ImdbDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, trained_vectorizer, torch_embeddings):
        self.file_list = file_list
        self.trained_vectorizer = trained_vectorizer
        self.torch_embedding = torch_embeddings

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        with open(self.file_list[idx], 'r') as f:
            content = f.read()
        cx = self.trained_vectorizer.transform([content]).tocoo()
        
        feature_list = []
        for i,j,v in zip(cx.row, cx.col, cx.data):
            feature_list += [j for _ in range(v)]
        feature_list = torch.LongTensor(feature_list).cuda()
        mean_embedding = torch.mean(self.torch_embedding(feature_list), dim=0)
        label = [1] if 'pos' in self.file_list[idx] else [0]
        return mean_embedding, torch.LongTensor(label)


