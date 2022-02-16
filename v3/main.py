import argparse
import os
import random
import shutil
import time
import warnings
import sys
import glob
import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter



from transformers import DistilBertTokenizer, DistilBertForMaskedLM



from fastCountVectorizer import FastCountVectorizer
from dan import DAN
from dataset import ReviewBatcher, ReviewsRaw



config = {
    'epochs': 5,
    'start_epoch': 0,
    'batch_size': 8,
    'lr': 0.0005,
    'dist_file': 'dist_file',
    'dist_backend': 'gloo',
    'intermediate_dimension': 1000,
    'embed_dimension': 1000,
    'data_dir': '../../split_reviews/reviews-*',
    'ngram_dir': '../data/fast_countvectorizer/top_ngrams_masked.txt',
    'batches_per_epoch': 2000,
}

def create_optimizers(model):
    dense_params, sparse_params = [], []
    #Extract the sparse parameters from the model
    for name, param in model.named_parameters():
        if 'embed' in name:
            sparse_params.append(param)
        else:
            dense_params.append(param)

    optimizer_sparse = torch.optim.SparseAdam(sparse_params, lr=config['lr'])
    optimizer_dense = torch.optim.Adam(dense_params, lr=config['lr'])
    return optimizer_sparse, optimizer_dense

def main():
    
    import os
    assert "SLURM_NPROCS" in os.environ

    config['num_procs'] = int(os.environ["SLURM_NPROCS"])
    config['proc_id'] = int(os.environ["SLURM_PROCID"])
    jobid = os.environ["SLURM_JOBID"]
    config['dist_url'] = "file://{}.{}".format(os.path.realpath(config['dist_file']), jobid)
    print("dist-url:{} at PROCID {} / {}".format(config['dist_url'], config['proc_id'], config['num_procs']))
    print(f"Initializing {config['proc_id']} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    ngpus_per_node = torch.cuda.device_count()
    config['world_size'] = ngpus_per_node * config['num_procs']
    
    # Use torch.multiprocessing.spawn to launch distributed processes
    print("World size: {}\nNgpus_per_node: {}".format(config['world_size'], ngpus_per_node))
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))

def cleanup():
    dist.destroy_process_group()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def main_worker(gpu, ngpus_per_node, config):
    
    # Unite Processes with init_process_group
    config['rank'] = config['proc_id'] * ngpus_per_node + gpu
    # setup(config['rank'], config['world_size'])
    dist.init_process_group(backend=config['dist_backend'], init_method=config['dist_url'],
                                world_size=config['world_size'], rank=config['rank'])
    print(f"Initialized {config['rank']} / {config['world_size']} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")



    
    ngram_tokenizer = FastCountVectorizer(config['ngram_dir'])

    #create model
    model = DAN(
        num_embeddings=ngram_tokenizer.size()+1,
        embed_dimension= config['embed_dimension'],
        intermediate_dimension= config['intermediate_dimension'],
        num_classes=30522,
    )

    model.to(gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids = [gpu])

    def criterion(input, target):
        return torch.mean(-torch.sum(target * torch.log(input), 1))

    optimizer_sparse, optimizer_dense = create_optimizers(model)

    # # TODO: potentially change this
    # cudnn.benchmark = True


    ###########################
    #   Data Loading
    filenames = glob.glob(config['data_dir'])
    
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    rawdataset = ReviewsRaw(filenames, ngram_tokenizer, tokenizer)
    reviewbatcher = ReviewBatcher(rawdataset, tokenizer, device=gpu, batch_size=config['batch_size'])
    ###########################


    optimizer_dense.zero_grad()
    optimizer_sparse.zero_grad()
    
    #tensorboard logging
    writer = SummaryWriter() if config['rank'] == 0 else None

    # distributed_tqdm = tqdm.tqdm if gpu == 0 else lambda x: x

    for epoch in range(config['start_epoch'], config['epochs']):
        if config['rank'] == 0:
            print("Epoch {}/{}".format(epoch, config['epochs']))
        
        for batch_index in range(config['batches_per_epoch']):
            start = time.time()
            inputs, labels = reviewbatcher.get_batch()
            ended_loading = time.time()

            inputs = inputs.to(gpu)
            labels = labels.to(gpu)

            outputs = model(inputs)

            optimizer_dense.zero_grad()
            optimizer_sparse.zero_grad()

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer_dense.step()
            optimizer_sparse.step()

            ended_step = time.time()

            if config['rank'] == 0:
                writer.add_scalar("Loss/train", loss, epoch * config['batches_per_epoch'] + batch_index)
                writer.add_scalar("loadingTime", ended_loading-start, epoch * config['batches_per_epoch'] + batch_index)
                writer.add_scalar("backpropTime", ended_step-ended_loading, epoch * config['batches_per_epoch'] + batch_index)
        
        #save model
        if config['rank'] == 0:
            os.makedirs("checkpoints", exist_ok=True)
            checkpoint_path =  f"checkpoints/model-{epoch}.checkpoint"
            torch.save(model.state_dict(), checkpoint_path)

    if config['rank'] == 0:
        writer.flush()
    cleanup()



if __name__ == '__main__':
    main()
