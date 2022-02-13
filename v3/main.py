import argparse
import os
import random
import shutil
import time
import warnings
import sys
import glob

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

from datasets.sentimentReviews import SentimentReviews
from fastCountVectorizer import FastCountVectorizer
from dan import DAN


from dataset import ReviewBatcher

config = {
    'epochs': 10,
    'start_epoch': 0,
    'batch_size': 512,
    'lr': 0.0005,
    'dist_file': 'dist_file',
    'dist_backend': 'gloo',
    'intermediate_dimension': 1000,
    'embed_dimension': 1000,
    'data_dir': '../../et_review_chunks',
    'ngrams': '../data/fast_countvectorizer/top_million_ngrams.txt'
}

def create_optimizers(model):
    dense_params, sparse_params = [], []
    #Extract the sparse parameters from the model
    for name, param in model.named_parameters():
        if 'embed' in name:
            sparse_params.append(param)
        else:
            dense_params.append(param)

    optimizer_sparse = torch.optim.SparseAdam(sparse_params, lr=args.lr)
    optimizer_dense = torch.optim.Adam(dense_params, lr=args.lr)
    return optimizer_sparse, optimizer_dense

def main():
    
    import os
    assert "SLURM_NPROCS" in os.environ

    config['num_procs'] = int(os.environ["SLURM_NPROCS"])
    config['proc_id'] = int(os.environ["SLURM_PROCID"])
    jobid = os.environ["SLURM_JOBID"]
    config['dist_url'] = "file://{}.{}".format(os.path.realpath(config['dist_file']), jobid)
    print("dist-url:{} at PROCID {} / {}".format(config['dist_url'], config['proc_id'], config['num_procs']))
    ngpus_per_node = torch.cuda.device_count()
    config['world_size'] = ngpus_per_node * config['num_procs']
    
    # Use torch.multiprocessing.spawn to launch distributed processes
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))

def cleanup():
    dist.destroy_process_group()

def main_worker(gpu, ngpus_per_node, config):
    
    # Unite Processes with init_process_group
    config['rank'] = config['proc_id'] * ngpus_per_node + gpu
    dist.init_process_group(backend=config['dist_backend'], init_method=config['dist_url'],
                                world_size=config['world_size'], rank=config['rank'])
    print(f"Initialized {config['rank']} / {config['world_size']}")



    # Load Tokenizer, Model, and Dataset
    masked_tokenizer = FastCountVectorizer(config['ngrams'])
    
    # create model
    model = DAN(
        num_embeddings=masked_tokenizer.size()+1,
        embed_dimension=args.embed_dimension,
        intermediate_dimension=args.intermediate_dimension
    )

    model.to(gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids = [gpu])

    criterion = torch.nn.NLLLoss()

    optimizer_sparse, optimizer_dense = create_optimizers(model)

    # TODO: potentially change this
    cudnn.benchmark = True


    ###########################
    #   Data Loading
    filenames = glob.glob("../data/sentiment_reviews/*")
    ngram_tokenizer = FastCountVectorizer("../data/fast_countvectorizer/top_ngrams_masked.txt")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    rawdataset = ReviewsRaw(filenames, ngram_tokenizer, tokenizer)
    reviewbatcher = ReviewBatcher(rawdataset, tokenizer)
    ###########################


    optimizer_dense.zero_grad()
    optimizer_sparse.zero_grad()
    
    inputs, labels = reviewbatcher.get_batch()
    inputs = inputs.to(gpu)
    labels = labels.to(gpu)
    outputs = model(inputs)

    loss_fn(outputs, labels).backward()

    optimizer_dense.step()
    optimizer_sparse.step()

    cleanup()





if __name__ == '__main__':
    main()
