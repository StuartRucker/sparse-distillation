from datasets import load_from_disk
import os
from transformers import BertTokenizer 
import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset, DataLoader


import wandb
import numpy as np

def init():
    config = {
        "name": "unigram",
        "description": "unigram model",
        "batch_size": 64,
        "total_iterations": 100,
        "learning_rate": 0.001,
        "log_frequency": 5,
    }
    
    wandb.init(project="Unigram", config=config)
        
    return config


class DAN(nn.Module):
    def __init__(self, embed_dimension=300, intermediate_dimension=512, num_classes=2):
        super(DAN, self).__init__()  
        #create embedding bag
        self.word_embed = nn.Embedding(num_classes, embed_dimension)
        self.position_embed = nn.Embedding(64, embed_dimension)

        self.fc1 = nn.Linear(embed_dimension, intermediate_dimension)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(intermediate_dimension, num_classes)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, words, positions):
        word_embeds = self.word_embed(words) # (batch_size, seq_len, embed_dim)
        position_embeds = self.position_embed(positions) # (batch_size, seq_len, embed_dim)
        
        # compute dot product of word embeddings and position embeddings
        
        dot_magnitudes = word_embeds * position_embeds # (batch_size, seq_len)
        

        

        summed_positions = torch.sum(dot_magnitudes, dim=1) # (batch_size, embed_dim)

        intermediate_output = self.fc1(summed_positions) # (batch_size, intermediate_dim)
        return self.fc2(self.relu(intermediate_output))


class UnigramDataset (torch.utils.data.Dataset):
    def __init__(self, mini=False):

        self.mini = mini

        self.dataset_books = load_from_disk(os.path.expanduser("~/hf_datasets/bookcorpus"))['train']
        self.dataset_wiki = load_from_disk(os.path.expanduser("~/hf_datasets/wikipedia"))['train']
        
        self.length = len(self.dataset_books) + len(self.dataset_wiki)
        
        tokenizer_path = os.path.join(os.path.dirname(__file__), "../data/bert_tokenizer")
        self.bert_tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

        self.MASK_ID = self.bert_tokenizer.convert_tokens_to_ids(["[MASK]"])[0]

        self.MAX_LENGTH = 32
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.mini:
            idx = idx % 50
        if idx < len(self.dataset_books):
            content = self.dataset_books[idx]['text']
        else:
            content = self.dataset_wiki[idx - len(self.dataset_books)]['text']

        tokenized_content = self.bert_tokenizer.tokenize(content, )[:self.MAX_LENGTH]
        original_tokenized_content_length = len(tokenized_content)
        while len(tokenized_content) < self.MAX_LENGTH:
            tokenized_content.append("[PAD]")

        # print(tokenized_content)

        # convert to ids
        token_ids = self.bert_tokenizer.convert_tokens_to_ids(tokenized_content)

        mask_idx = random.randint(0, original_tokenized_content_length - 1)
        relative_positions = [32+i-mask_idx for i in range(len(token_ids))]
        masked_id = token_ids[mask_idx]
        token_ids[mask_idx] = self.MASK_ID

        return torch.LongTensor(token_ids), torch.LongTensor(relative_positions), torch.LongTensor([masked_id])

        # get the bert index of the masked_word
        

def main():

    config = init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = UnigramDataset()
    model = DAN(num_classes=dataset.bert_tokenizer.vocab_size).to(device)
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    iteration_cnt = 0
    criterion = torch.nn.CrossEntropyLoss()

    for words, positions, label in data_loader:
        if iteration_cnt >= config['total_iterations']:
            break
        # move words to device
        words, positions, label = words.to(device), positions.to(device), label.to(device)

        optimizer.zero_grad()

        output = model(words, positions)
        loss = criterion(output, label.flatten())
        loss.backward()
        optimizer.step()

        iteration_cnt += 1
        print(f"Iteration {iteration_cnt}: {loss.item()}")
        wandb.log({
            "loss": loss.item(),
            "iteration": iteration_cnt,
        })
        # if iteration_cnt % config['log_frequency'] == 0:
        #     # log position embeddings of the model
    labels = np.array([32-i for i in range(64)])
    embeddings = model.position_embed.weight.detach().cpu().numpy()

    # append labels as the last column of the embeddings
    labeled_embeddings = np.concatenate((embeddings, labels.reshape(-1, 1)), axis=1)

    wandb.log({
        "embeddings": wandb.Table(
            columns =  [f"d{i}" for i in range(300)] + ['a_label'],
            data    = labeled_embeddings
        )
    })
    torch.save(model.state_dict(), f"unigram_model.pt")
    

if __name__ == '__main__':
    main()





