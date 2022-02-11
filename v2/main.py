from torch.utils.data import DataLoader
import time
import torch
import argparse
from tqdm import tqdm
import os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import horovod.torch as hvd

from datasets.sentimentReviews import SentimentReviews
from fastCountVectorizer import FastCountVectorizer
from model import DAN
from util import load_model, save_model, check_accuracy
import itertools

config = {
        'learning_rate': .0005,
        'batch_size': 1024,
        'epochs': 50,
        'intermediate_dimension':1000,
        'embed_dimension':1000
        }

def parse_args():
    parser = argparse.ArgumentParser(description='Sparse Distillation DAN')
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoints')
    parser.add_argument('--data_dir', type=str, default='../../medium_et_review_chunks')
    parser.add_argument('--ngrams', type=str, default='../data/fast_countvectorizer/top_million_ngrams.txt')

    return parser.parse_args()



def main():
    args = parse_args()
    use_hvd = False
    if use_hvd:
        hvd.init()
    if not use_hvd or hvd.rank()==0:
        writer = SummaryWriter()
    print(config) 
    data_paths = [os.path.join(args.data_dir,f) for f in os.listdir(args.data_dir)]

    
    tokenizer = FastCountVectorizer(args.ngrams)
    print('Initialized Tokenizer')
    entire_dataset = SentimentReviews(data_paths, tokenizer)
    train_amount = int(len(entire_dataset)*.8)
    dataset, test_dataset = torch.utils.data.random_split(entire_dataset, [train_amount, len(entire_dataset)-train_amount])
    print("Initialized Dataset")
    
    

    model = DAN(
        num_embeddings=tokenizer.size()+1,
        embed_dimension=config['embed_dimension'],
        intermediate_dimension=config['intermediate_dimension']
    )
    

    if use_hvd and torch.cuda.is_available():
        torch.cuda.set_device(hvd.local_rank())
        
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

   # model.to(device)
    sparse_params, sparse_named_params = [],[]
    dense_params, dense_named_params = [],[]
    for _name, _data in model.named_parameters():
        if 'embed' in _name:
            sparse_params.append(_data)
            sparse_named_params.append((_name, _data))
        else:
            dense_params.append(_data)
            dense_named_params.append((_name, _data))
    model.cuda()
     

    loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
    
    opt_sparse = torch.optim.SparseAdam(sparse_params, lr=config['learning_rate'])
    opt_dense = torch.optim.Adam(dense_params, lr=config['learning_rate'])

    if use_hvd:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

        hvd.broadcast_optimizer_state(opt_sparse, root_rank=0)
        hvd.broadcast_optimizer_state(opt_dense, root_rank=0)
        opt_sparse = hvd.DistributedOptimizer(opt_sparse, named_parameters=sparse_named_params)
        opt_dense = hvd.DistributedOptimizer(opt_dense, named_parameters=dense_named_params)
   
    starting_epoch=0
    sampler_kwargs = kwargs = {'num_workers': 1, 'pin_memory': False, 'batch_size':config['batch_size']}
    print("starting training")
    for epoch in range(starting_epoch, config['epochs']):
        print(f'Epoch {epoch}')
        end = time.time()
        for train_features, train_labels in tqdm(DataLoader(dataset,  **sampler_kwargs)):
            opt_sparse.zero_grad()
            opt_dense.zero_grad()
            output = model(train_features.cuda())
            loss = loss_fn(output, train_labels.cuda())
            
            loss.backward()
            opt_sparse.step()
            opt_dense.step()
            print(time.time()-end)
            end = time.time()
        print(f"Loss {loss.cpu()}")
        if not use_hvd or hvd.rank()==0:
            save_model(model, opt_sparse, opt_dense, epoch, args.checkpoint_dir)

if __name__ == '__main__':
    main()


