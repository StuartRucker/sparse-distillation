import argparse

from tokenizer import Tokenizer
from dan import train_model, train_mask_model, DAN
import torch
import wandb

from config import config

from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='V4')
    parser.add_argument('--corpus', type=str, default=None, help='Corpus [Amazon]')
    parser.add_argument('--d_train', type=str, default='IMDB_train', help='D_train [IMDB]')
    parser.add_argument('--d_test', type=str, default='IMDB_test', help='D_train D_val [IMDB]')
    parser.add_argument('--pretrain', action='store_true', help="Self pretrain for masked word prediction")
    parser.add_argument('--kd', action='store_true', help="Use Knowledge Distillation")
    parser.add_argument('--ft', action='store_true', help="Use Fine Tuning")
    parser.add_argument('--mini', action='store_true', help="Scale things down for testing")
    parser.add_argument('--from_scratch', action='store_true', help="Scale things down for testing")
    return parser.parse_args()

def time():
    now = datetime.now() # current date and time
    return now.strftime("%H:%M:%S")

def main(args):

    run_name = f'[Corpus:{args.corpus} Dtrain:{args.d_train} pretrain:{args.pretrain} kd:{args.kd} fd:{args.ft} mini:{args.mini}]'
    print(f'Running... {run_name}')

    print(f"[{time()}] Loading Tokenizer...")
    tokenizer = Tokenizer(args.corpus, args.d_train, mini=args.mini)

    # combine args and config into one
    config['corpus'] = args.corpus
    config['d_train'] = args.d_train
    config['d_test'] = args.d_test
    config['pretrain'] = args.pretrain
    config['kd'] = args.kd
    config['ft'] = args.ft
    config['mini'] = args.mini

    wandb.init(project="DAN",
        config = config,
    )

    
    saved_weights = None
    if args.pretrain:
        tokenizer.transform(["[MASK]"], mask=True) # put the tokenizer in mask mode
        mask_model = DAN(num_classes=tokenizer.get_bert_vocabulary_size(), intermediate_dimension=config['pretrain_intermediate_dimension'], 
                num_embeddings=len(tokenizer)+1)
        train_mask_model(mask_model, tokenizer, args.corpus, mini=args.mini, run_name='pretrain:'+run_name, from_scratch=args.from_scratch)
        
        #transfer weights from mask model to model
        embedding_weights = mask_model.embed.weight.data.clone()
        
        
        print("Moving Model weights to later model...")
        tokenizer.transform(["[MASK]"], mask=False) # put the tokenizer in mnormal mode
        saved_weights = torch.cat([embedding_weights[:len(tokenizer)], embedding_weights[[-1]]], dim=0)
        del mask_model
        
    model = DAN(num_embeddings=len(tokenizer)+1)
    if saved_weights is not None:
        model.embed.weight.data = saved_weights
    
    if args.kd:
        print("TODO: Fine tune teacher model")
        pass
    
    if args.ft:
        print(f"[{time()}] Fine Tuning on {args.d_train}")
        train_model(model, tokenizer, args.d_train, args.d_test, mini=args.mini, run_name=run_name, from_scratch=args.pretrain or args.from_scratch)



if __name__ == '__main__':
    args = parse_args()
    main(args)