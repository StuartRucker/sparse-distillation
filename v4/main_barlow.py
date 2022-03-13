import argparse

from tokenizer import Tokenizer
from dan import train_model, train_mask_model, DAN, CBOW
from barlow import BARLOW
import torch
import wandb
from data import get_barlow_dataset
from torch.utils.data import Dataset, DataLoader

from config import config as mega_config
import torch.nn as nn

from main import fix_print
fix_print()

from datetime import datetime

config = {
    'mini': True,
    'learning_rate': 0.001,
    'log_every': 100,
    'embed_dimension': 1000,
    'output_dimension': 40,
    'max_iterations': 40000,
    'batch_size': 256,
    'lambda': 0.01,

}
if config['mini']:
    config['log_every'] = 10
    config['max_iterations'] =  100
    config['batch_size'] =  20
for key, value in mega_config.items():
    if key.startswith("finetune"):
        config[key] = value

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def barlow_loss(z_a, z_b):
    # normalize repr. along the batch dimension
    z_a_norm = (z_a - torch.mean(z_a, dim=0)) / torch.std(z_a,dim=0) # NxD
    z_b_norm = (z_b - torch.mean(z_b, dim=0)) / torch.std(z_b,dim=0) # NxD

    # cross-correlation matrix
    c = torch.mm(z_a_norm.t(), z_b_norm) / z_a_norm.shape[0] # DxD

    # loss
    # c_diff = (c - eye(D)).pow(2) # DxD
    c_diff = (c - torch.eye(c.shape[0], device=device)).pow(2) # DxD
    # multiply off-diagonal elems of c_diff by lambda
    # off_diagonal(c_diff).mul_(lambda)
    mult_values = torch.ones(c_diff.shape, device=device)*config['lambda'] + torch.eye(c_diff.shape[0],device=device)*(1-config['lambda'])   
    c_diff = c_diff * mult_values
    loss = c_diff.sum()
    return loss

wandb.init(project="Barlow", config=config)


print("About to Create English Tokenizer...")
tokenizer_en = Tokenizer('wmt14_en_train', 'IMDB_train', mini=config['mini'])
print("About to Create Foreign Tokenizer...")
tokenizer_foreign = Tokenizer('wmt14_de_train', None, mini=config['mini'])

num_embeddings_en = len(tokenizer_en) + 1
num_embeddings_foreign = len(tokenizer_foreign) + 1

embed_layer_en = nn.EmbeddingBag(num_embeddings_en, embedding_dim=config['embed_dimension'], mode='mean', sparse=True, padding_idx=num_embeddings_en-1)
embed_layer_foreign = nn.EmbeddingBag(num_embeddings_foreign, embedding_dim=config['embed_dimension'], mode='mean', sparse=True, padding_idx=num_embeddings_foreign-1)


barlow_en = BARLOW(embed_layer_en, embed_dimension=config['embed_dimension'], num_embeddings=num_embeddings_en, output_dimension=config['output_dimension'])
barlow_foreign = BARLOW(embed_layer_foreign, embed_dimension=config['embed_dimension'], num_embeddings=num_embeddings_foreign, output_dimension=config['output_dimension'])


barlow_en.to(device)
barlow_foreign.to(device)
embed_layer_en.to(device)
embed_layer_foreign.to(device)


dense_params = [param for name, param in barlow_en.named_parameters() if 'embed' not in name.lower()]
sparse_params = [param for name, param in barlow_en.named_parameters() if 'embed'  in name.lower()]

dense_params += [param for name, param in barlow_foreign.named_parameters() if 'embed' not in name.lower()]
sparse_params += [param for name, param in barlow_foreign.named_parameters() if 'embed'  in name.lower()]


optimizer = torch.optim.Adam(dense_params, lr=config['learning_rate'])
optimizer_sparse = torch.optim.SparseAdam( sparse_params, lr=config['learning_rate'])


train_dataset = get_barlow_dataset(tokenizer_en, tokenizer_foreign, mini=config['mini'])
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

iteration_cnt = 0
for epoch in range(20):
    if iteration_cnt >= config['max_iterations']:
            break
    for data_en, data_foreign in train_loader:
        if iteration_cnt >= config['max_iterations']:
            break
    
        data_en, data_foreign = data_en.to(device), data_foreign.to(device)


        optimizer.zero_grad()
        optimizer_sparse.zero_grad()
        
        output_en, output_foreign = barlow_en(data_en), barlow_foreign(data_foreign)
        loss = barlow_loss(output_en, output_foreign)

        
        loss.backward()
        optimizer.step()
        optimizer_sparse.step()
        
        if iteration_cnt % config['log_every'] == 0:
            print("Iteration: ", iteration_cnt, " Loss: ", loss.item())
            
        wandb.log({'Pretrain/Loss/train': loss.item(), 'Pretrain/iteration': iteration_cnt})
        iteration_cnt += 1

print("Finished Pretraining Using Barlow")
saved_weights = embed_layer_en.weight.data.clone()

#most likely done by garbage control
del embed_layer_en
del embed_layer_foreign
del barlow_en
del barlow_foreign
del optimizer
del optimizer_sparse
#Fine Tuning

model = DAN(num_embeddings = num_embeddings_en)
model.embed.weight.data = saved_weights

train_model(model, tokenizer_en, "IMDB_train", 'IMDB_test', mini=config['mini'], run_name="Barlow Pretrained", from_scratch=True)