import glob
import random
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



eps = lambda: random.random()*.01
MAX_WORDS_TO_PAD_TO = 300
NUM_WORDS_TO_TRACK = 100

# sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")

# #next get the train y
# train_y = []
# for train_file_s in train_files:
#     with open(train_file_s, "r") as train_file:
#         train_y.append(sentiment_analysis(train_file.read()))

vectorizer = CountVectorizer(ngram_range=(1, 4), max_features=NUM_WORDS_TO_TRACK, input='filename')

train_files = random.sample(glob.glob("data/imdb/train/*/*.txt"), 5000)
test_files = random.sample(glob.glob("data/imdb/test/*/*.txt"), 5000)
# train_files = glob.glob("data/imdb/train/*/*.txt")
# test_files = glob.glob("data/imdb/test/*/*.txt")



trained_vectorizer = vectorizer.fit(train_files)

no_word_token = len(trained_vectorizer.get_feature_names_out())
num_embeddings = no_word_token+1





class ImdbDataset(Dataset):
    def __init__(self, file_list, trained_vectorizer):
        self.file_list = file_list
        self.trained_vectorizer = trained_vectorizer

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        
        
        cx = self.trained_vectorizer.transform([self.file_list[idx]]).tocoo()
        feature_list = []
        for i,j,v in zip(cx.row, cx.col, cx.data):
            feature_list += [j for _ in range(v)]
        if len(feature_list) > MAX_WORDS_TO_PAD_TO:
            feature_list = random.sample(feature_list, MAX_WORDS_TO_PAD_TO)
        while len(feature_list) < MAX_WORDS_TO_PAD_TO:
            feature_list.append(no_word_token)
        label = [eps(),1-eps()] if 'pos' in self.file_list[idx] else [1-eps(),eps()]
        return torch.LongTensor(feature_list), torch.FloatTensor(label)



train_dataset = ImdbDataset(train_files, trained_vectorizer)
test_dataset = ImdbDataset(test_files, trained_vectorizer)



embed_dimension = 10

train_y = [[0.0,1.0] if 'pos'  in filename else [1.0,0.0] for filename in train_files]




class DAN(nn.Module):
    def __init__(self):
        super(DAN, self).__init__()
        self.embed = nn.EmbeddingBag(num_embeddings, embed_dimension, mode='mean', padding_idx=no_word_token)
        self.fc1 = nn.Linear(embed_dimension, embed_dimension)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(embed_dimension, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embedded = self.embed(x)
        inter1 = self.relu(self.fc1(embedded))
        return self.softmax(self.fc2(inter1))


model = DAN()

def save_model(model, optimizer, epoch):
    PATH = "model.pt"
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, PATH)
def load_model(model, optimizer):
    PATH = "model.pt"
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.train()
    return epoch

loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    answer = -1
    with torch.no_grad():
        for x, y in loader:

            scores = model(x)
            
            _, predictions = scores.max(1)
            _, true_value = y.max(1)
            num_correct += (predictions == true_value).sum()
            num_samples += predictions.size(0)
        
        answer = float(num_correct)/float(num_samples)
    
    model.train()
    return answer


for epoch in range(3):
    load_model(model, optimizer)
    for train_features, train_labels in tqdm(train_dataloader):
        optimizer.zero_grad()
        output = model(train_features)
        loss = loss_fn(output, train_labels)
        loss.backward()
        optimizer.step()
    
    save_model(model, optimizer, epoch)
    if epoch % 5 == 0:
        print(f'train accuracy {check_accuracy(train_dataloader, model)}')
        print(f'test accuracy {check_accuracy(test_dataloader, model)}')
    
    print(float(loss))