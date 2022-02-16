import torch
import torch.nn as nn
import torch.nn.functional as F


class DAN(nn.Module):
    def __init__(self, num_embeddings, embed_dimension=1000, intermediate_dimension=1000, no_word_token=0, num_classes=2):
        super(DAN, self).__init__()
        
        self.embed = nn.EmbeddingBag(num_embeddings, embed_dimension, sparse=True, mode='sum')
        self.fc1 = nn.Linear(embed_dimension, embed_dimension)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(embed_dimension, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embedded = self.embed(x)/x.size(dim=1)
        inter1 = self.relu(self.fc1(embedded))
        return self.softmax(self.fc2(inter1))
