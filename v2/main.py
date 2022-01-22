from torch.utils.data import DataLoader
import torch
import argparse
from tqdm import tqdm
import os
from pathlib import Path

from datasets.randomReviews import RandomReviews
from fastCountVectorizer import FastCountVectorizer
from model import DAN
from util import load_model, save_model, check_accuracy



def parse_args():
    parser = argparse.ArgumentParser(description='Sparse Distillation DAN')
    parser.add_argument('-checkpoint_dir', type=str, default='../checkpoints')
    parser.add_argument('-data_dir', type=str, default='../data/sentiment_reviews_shuffled')
    parser.add_argument('-ngrams', type=str, default='../data/fast_countvectorizer/top_million_ngrams.txt')

    return parser.parse_args()



def main():
    args = parse_args()

    data_paths = [os.path.join(args.data_dir,f) for f in os.listdir(args.data_dir)]

    
    tokenizer = FastCountVectorizer(args.ngrams)
    print('Initialized Tokenizer')
    dataset = lambda: RandomReviews(data_paths[:-1], tokenizer)
    test_dataset = lambda: RandomReviews([data_paths[-1]], tokenizer)
    print("Initialized Dataset")
    
    

    model = DAN(
        num_embeddings=tokenizer.size()+1,
        embed_dimension=10,
        intermediate_dimension=10
    )

    


    loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)

    starting_epoch = load_model(model, optimizer, args.checkpoint_dir)+1


    for epoch in range(starting_epoch, 5):
        print(f'Epoch {epoch}')
        for train_features, train_labels in tqdm(DataLoader(dataset(), batch_size=16)):
            optimizer.zero_grad()
            output = model(train_features)
            loss = loss_fn(output, train_labels)
            loss.backward()
            optimizer.step()

        # save_model(model, optimizer, epoch, args.checkpoint_dir)
        print(f"train accuracy {check_accuracy(DataLoader(dataset()), model)}")
        print(f"test accuracy {check_accuracy(DataLoader(test_dataset()), model)}")



if __name__ == '__main__':
    main()


