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

import wandb

from data import get_ft_dataset, get_pretrain_dataset



class CBOW(nn.Module):
    def __init__(self, embed_dimension=1000, intermediate_dimension=1000, num_embeddings=1000000, pad_dimension=1, num_classes=2):
        super(CBOW, self).__init__()  
        #create embedding bag
        self.pad_dimension = pad_dimension
        self.embed = nn.EmbeddingBag(num_embeddings, embedding_dim=embed_dimension, mode='mean', sparse=True, padding_idx=num_embeddings-1)
        self.fc1 = nn.Linear(embed_dimension, num_classes)
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(intermediate_dimension, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, data):
        before_words = data[:,0]
        after_words = data[:,1]

        embed_before =  self.embed(before_words)
        embed_after =  self.embed(after_words)
        concatenated_embeddings = embed_before+embed_after#torch.cat([embed_before, embed_after], dim=1)
        # inter1 = self.relu(self.fc1(concatenated_embeddings))
        # return self.softmax(self.fc2(inter1))
        return self.softmax(self.fc1(concatenated_embeddings))

class DAN(nn.Module):
    def __init__(self, embed_dimension=1000, intermediate_dimension=1000, num_embeddings=1000000, num_classes=2,):
        super(DAN, self).__init__()  
        #create embedding bag
        self.embed = nn.EmbeddingBag(num_embeddings, embedding_dim=embed_dimension, mode='mean', sparse=True, padding_idx=num_embeddings-1)
        self.fc1 = nn.Linear(embed_dimension, intermediate_dimension)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(intermediate_dimension, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embed =  self.embed(x)
        inter1 = self.relu(self.fc1(embed))
        return self.softmax(self.fc2(inter1))


def load_model(model, run_name):
    directory_path = os.path.join(os.path.dirname(__file__), f"cached/models/{run_name}")
    if not os.path.exists(directory_path):
        return -1
    files = os.listdir(directory_path)
    files.sort(key=lambda x: int(x.split('.')[0]))
    if len(files) > 0:
        print(f"Loading Cached Model {files[-1]}")
        filepath = os.path.join(directory_path, files[-1])
        loaded = torch.load(filepath)
        model.load_state_dict(loaded['model'])
        return int(files[-1].split('.')[0])
    else:
        print("Starting Training From Scratch...")
        return -1


def save_model(model, run_name, epoch):
    #make directory if it doesn't exist
    directory_path = os.path.join(os.path.dirname(__file__), f"cached/models/{run_name}")
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    filepath = os.path.join(directory_path, f'{epoch}.model')
    
    torch.save({'model': model.state_dict()}, filepath)


def eval_model(model, data_loader, device, limit=100000):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target.flatten()).sum().item()
            if total >= limit:
                break
    model.train()
    return correct/total

def train_mask_model(mask_model, pretrain_model, tokenizer, corpus, mini=False, run_name='unnamed', from_scratch=False):
    assert 'pretrain' in run_name.lower()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_model.to(device)
    

    start_iteration_cnt = load_model(mask_model, run_name) if not from_scratch else -1

    
    dense_params = [param for name, param in mask_model.named_parameters() if 'embed' not in name.lower()]
    sparse_params = [param for name, param in mask_model.named_parameters() if 'embed'  in name.lower()]
    optimizer = torch.optim.Adam(dense_params, lr=config['pretrain_learning_rate'])
    optimizer_sparse = torch.optim.SparseAdam( sparse_params, lr=config['pretrain_learning_rate'])

    print("Corpus: ", corpus)
    train_dataset = get_pretrain_dataset(corpus, tokenizer, mini=mini, mode=pretrain_model)
    train_loader = DataLoader(train_dataset, batch_size=config['pretrain_batch_size'], shuffle=True, num_workers=4, pin_memory=True)

    
    mask_model.train()
    criterion = torch.nn.CrossEntropyLoss()

    iteration_cnt = start_iteration_cnt
    for epoch in range(config['pretrain_max_epochs']+1):


        for data, target in train_loader:
            if iteration_cnt >= config['pretrain_max_iterations']:
                return
            
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            optimizer_sparse.zero_grad()
            
            output = mask_model(data)

            loss = criterion(output, target.flatten())
            
            loss.backward()
            optimizer.step()
            optimizer_sparse.step()
            
            if iteration_cnt % config['pretrain_log_every'] == 0:
                wandb.log({'Pretrain/Loss/train': loss.item(), 'Pretrain/iteration': iteration_cnt})
                print("Iteration: ", iteration_cnt, " Loss: ", loss.item())
                
            if iteration_cnt % config['pretrain_save_model_every'] == 0:
                save_model(mask_model, run_name, iteration_cnt)
            
            iteration_cnt += 1

    
        



def train_model(model, tokenizer, d_train, d_test, mini=False, run_name='unnamed', from_scratch=False):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
     
    start_epoch = load_model(model, run_name) if not from_scratch else -1

    
    dense_params = [param for name, param in model.named_parameters() if 'embed' not in name.lower()]
    sparse_params = [param for name, param in model.named_parameters() if 'embed'  in name.lower()]
    optimizer = torch.optim.Adam(dense_params, lr=config['finetune_learning_rate'])
    optimizer_sparse = torch.optim.SparseAdam( sparse_params, lr=config['finetune_learning_rate'])

    train_dataset = get_ft_dataset(d_train, tokenizer, mini)
    train_loader = DataLoader(train_dataset, batch_size=config['finetune_batch_size'], shuffle=True, num_workers=4, pin_memory=True)

    test_dataset = get_ft_dataset(d_test, tokenizer, mini)
    test_loader = DataLoader(test_dataset, batch_size=config['finetune_batch_size'], shuffle=True, num_workers=4, pin_memory=True)

    model.train()
    criterion = torch.nn.CrossEntropyLoss()


    for epoch in range(start_epoch+1, min(start_epoch+config['finetune_epochs']+1, config['finetune_max_epochs']+1)):
        start_time_epoch = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            optimizer_sparse.zero_grad()
            

            output = model(data)


            loss = criterion(output, target.flatten())
            
            
            loss.backward()
            optimizer.step()
            optimizer_sparse.step()


            if batch_idx % 10 == 0:
                # writer.add_scalar('Loss/train', loss.item(), epoch*len(train_loader)+batch_idx)
                wandb.log({'Finetune/Loss/train': loss.item(), 'Finetune/iteration': epoch*len(train_loader)+batch_idx})
        

        wandb.log({'Finetune/Accuracy/train': eval_model(model, train_loader, device, limit=1000),
            'Finetune/Accuracy/test': eval_model(model, test_loader, device, limit=1000),
            'Finetune/iteration': (1+epoch)*len(train_loader)
        })
        
        
        print(f"Epoch: {epoch} Loss: {loss}")

        #save the model
        save_model(model, run_name, epoch)



