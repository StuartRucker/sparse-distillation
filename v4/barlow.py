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



class BARLOW(nn.Module):
    def __init__(self, embed_layer, embed_dimension=1000, num_embeddings=1000000, output_dimension=2):
        super(BARLOW, self).__init__()  
    
        
        self.fc1 = nn.Linear(embed_dimension, output_dimension)
        self.embed = embed_layer
        self.tanh = nn.Tanh() #dim = 1

    def forward(self, data):
        embeddings =  self.embed(data)
        return self.tanh(self.fc1(embeddings))


