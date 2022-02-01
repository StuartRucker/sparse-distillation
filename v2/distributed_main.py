
from datasets.sentimentReviews import SentimentReviews
from fastCountVectorizer import FastCountVectorizer


config = {
    'learning_rate': .005,
    'batch_size': 8,
    'epochs': 50,
    'intermediate_dimension':1000,
    'embed_dimension':1000
}
import os
import builtins
import argparse
import torch
import numpy as np 
import random
import torch.distributed as dist

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size per GPU')
    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, 
                        help='start epoch number (useful on restarts)')
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    # DDP configs:
    parser.add_argument('--world-size', default=-1, type=int, 
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, 
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str, 
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, 
                        help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int, 
                        help='local rank for distributed training')
    parser.add_argument('--workers', default=-1, type=int, 
                        help='number of data loading workers')

    parser.add_argument('--ngrams', type=str, default='../data/fast_countvectorizer/top_million_ngrams.txt')
    args = parser.parse_args()
    return args
                                         
def main(args):
    # DDP setting
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()

    if args.distributed:
        if args.local_rank != -1: # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # suppress printing if not on master gpu
    if args.rank!=0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    
    #Initialize Tokenizer
    tokenizer = FastCountVectorizer(args.ngrams)

    ### model ###
    model = DAN(
        num_embeddings=tokenizer.size()+1,
        embed_dimension=config['embed_dimension'],
        intermediate_dimension=config['intermediate_dimension']
    )

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
            model_without_ddp = model.module
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")
        
    ### optimizer ###
    sparse_params, sparse_named_params = [],[]
    dense_params, dense_named_params = [],[]
    for _name, _data in model.named_parameters():
        if 'embed' in _name:
            sparse_params.append(_data)
            sparse_named_params.append((_name, _data))
        else:
            dense_params.append(_data)
            dense_named_params.append((_name, _data))
    optimizer_dense = torch.optim.Adam(dense_params, lr=args.lr, weight_decay=1e-5)
    optimizer_sparse = torch.optim.SparseAdam(sparse_params, lr=args.lr, weight_decay=1e-5)
    
    ### resume training if necessary ###
    if args.resume:
        pass
    
    ### data ###
    entire_dataset = SentimentReviews(data_paths, tokenizer)
    train_amount = int(len(entire_dataset)*.8)
    train_dataset, val_dataset = torch.utils.data.random_split(entire_dataset, [train_amount, len(entire_dataset)-train_amount])

    
    train_sampler = data.distributed.DistributedSampler(dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=True)
    
    torch.backends.cudnn.benchmark = True
    
    criterion = torch.nn.KLDivLoss(reduction='batchmean')

    ### main loop ###
    for epoch in range(args.start_epoch, args.epochs):
        np.random.seed(epoch)
        random.seed(epoch)
        # fix sampling seed such that each gpu gets different part of dataset
        if args.distributed: 
            train_loader.sampler.set_epoch(epoch)
        
        # adjust lr if needed #
        
        train_one_epoch(train_loader, model, criterion, optimizer, epoch, args)
        if args.rank == 0: # only val and save on master node
            validate(val_loader, model, criterion, epoch, args)
            # save checkpoint if needed #

def train_one_epoch(train_loader, model, criterion, optimizer_sparse, optimizer_dense, epoch, args):
    for train_features, train_labels in train_loader:
        optimizer_sparse.zero_grad()
        optimizer_dense.zero_grad()
        output = model(train_features.cuda())
        loss = loss_fn(output, train_labels.cuda())
        
        # writer.add_scalar("Loss/train", loss, epoch)
        loss.backward()
        optimizer_sparse.step()
        optimizer_dense.step()

def validate(val_loader, model, criterion, epoch, args):
    pass

if __name__ == '__main__':
    args = parse_args()
    main(args)