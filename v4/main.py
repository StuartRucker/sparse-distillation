import argparse

from tokenizer import Tokenizer
from dan import train_model, DAN
import torch

from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='V4')
    parser.add_argument('--corpus', type=str, default=None, help='Corpus [Amazon]')
    parser.add_argument('--d_train', type=str, default='IMDB_train', help='D_train [IMDB]')
    parser.add_argument('--d_test', type=str, default='IMDB_test', help='D_train D_val [IMDB]')
    parser.add_argument('--kd', action='store_true', help="Use Knowledge Distillation")
    parser.add_argument('--ft', action='store_true', help="Use Fine Tuning")
    parser.add_argument('--mini', action='store_true', help="Scale things down for testing")
    return parser.parse_args()

def time():
    now = datetime.now() # current date and time
    return now.strftime("%H:%M:%S")

def main(args):

    run_name = f'[Corpus:{args.corpus} Dtrain:{args.d_train} kd:{args.kd} fd:{args.ft} mini:{args.mini}]'
    print(f'Running... {run_name}')

    print(f"[{time()}] Loading Tokenizer...")
    tokenizer = Tokenizer(args.corpus, args.d_train, mini=args.mini)


    model = DAN()
    embeddings = torch.nn.Embedding(len(tokenizer.countvectorizer.get_feature_names()), 1000, sparse=True)
    if args.kd:
        print("TODO: Fine tune teacher model")
        pass
    
    if args.ft:
        print(f"[{time()}] Fine Tuning on {args.d_train}")
        train_model(model, embeddings, tokenizer, args.d_train, args.d_test, mini=args.mini, run_name=run_name)



if __name__ == '__main__':
    args = parse_args()
    main(args)