from cProfile import run
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import tqdm
from config import config
import os
import time

from data import get_ft_dataset

from torch.utils.tensorboard import SummaryWriter


class DAN(nn.Module):
    def __init__(self, embed_dimension=1000, intermediate_dimension=1000, no_word_token=0, num_classes=2):
        super(DAN, self).__init__()        
        self.fc1 = nn.Linear(embed_dimension, intermediate_dimension)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(intermediate_dimension, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        inter1 = self.relu(self.fc1(x))
        return self.softmax(self.fc2(inter1))


def load_model(model, embeddings, run_name):
    directory_path = os.path.join(os.path.dirname(__file__), f"cached/models/{run_name}")
    if not os.path.exists(directory_path):
        return
    files = os.listdir(directory_path)
    files.sort()
    if len(files) > 0:
        print(f"Loading Cached Model {files[-1]}")
        filepath = os.path.join(directory_path, files[-1])
        loaded = torch.load(filepath)
        model.load_state_dict(loaded['model'])
        embeddings.weight = loaded['embeddings']
    else:
        print("Starting Training From Scratch...")


def save_model(model, embeddings, run_name, epoch):
    #make directory if it doesn't exist
    directory_path = os.path.join(os.path.dirname(__file__), f"cached/models/{run_name}")
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    filepath = os.path.join(directory_path, f'{epoch}.model')
    
    torch.save({'model': model.state_dict(), 'embeddings': embeddings.weight}, filepath)


def eval_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct/total

def train_model(model, embeddings, tokenizer, d_train, d_test, mini=False, run_name=''):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    embeddings.to(device)    
    writer = SummaryWriter(comment=run_name)

    load_model(model, embeddings, run_name)

    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    optimizer_sparse = torch.optim.SparseAdam( [embeddings.weight], lr=config['learning_rate'])

    train_dataset = get_ft_dataset(d_train, tokenizer, embeddings, mini)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    test_dataset = get_ft_dataset(d_test, tokenizer, embeddings, mini)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True)

    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(config['epochs']):
        start_time_epoch = time.time()
        for batch_idx, (data, target) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            optimizer_sparse.zero_grad()
            
            output = model(data)


            loss = criterion(output, target.flatten())
            
            
            loss.backward()
            optimizer.step()
            optimizer_sparse.step()


            if batch_idx % 10 == 0:
                writer.add_scalar('Loss/train', loss.item(), epoch*len(train_loader)+batch_idx)
        
        writer.add_scalar('Time/epoch', time.time()-start_time_epoch, epoch)
        eval_start_time = time.time()
        writer.add_scalar('Accuracy/train', eval_model(model, train_loader, device), epoch*len(train_loader)+batch_idx)
        writer.add_scalar('Accuracy/test', eval_model(model, test_loader, device), epoch*len(test_loader)+batch_idx)
        writer.add_scalar('Time/eval', time.time()-eval_start_time, epoch)
        
        
        print(f"Epoch: {epoch} Loss: {loss}")

        #save the model
        save_model(model, embeddings, run_name, epoch)



