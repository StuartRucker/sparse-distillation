import re
import glob
import os
import torch

def extract_ngrams(text, max_range=4):
    all_words = re.findall( r'(?u)\b\w\w+\b', text.lower())
    ngrams = []
    for i in range(len(all_words)):
        for j in range(i+1, 1+min(i+max_range, len(all_words))):
            ngrams.append(' '.join(all_words[i:j]))
    return ngrams

def get_epoch_from_str(s):
    start_index = s.rfind("-")+1
    end_index = s.rfind(".")
    return s[start_index:end_index]

def load_model(model, opt_sparse, opt_dense, folder):
    potential_models = glob.glob(os.path.join(folder + "/model-*.pt"))
    if len(potential_models) == 0:
        print("Starting From Scratch - Found No Checkpoint")
        return -1
    
    model_path = max(potential_models, key=get_epoch_from_str)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    opt_sparse.load_state_dict(checkpoint['opt_sparse_state_dict'])
    opt_dense.load_state_dict(checkpoint['opt_dense_state_dict'])
    model.train()
    print(f"Found Saved Checkpoint, starting from epoch {checkpoint['epoch']}")
    return checkpoint['epoch']

def save_model(model, opt_sparse, opt_dense, epoch, folder):
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'opt_sparse_state_dict': opt_sparse.state_dict(),
                'opt_dense_state_dict': opt_dense.state_dict()
                }, os.path.join(folder, f"model-{epoch}.pt"))


def check_accuracy(loader, model, device, first_n_samples=100):
    num_correct = 0
    num_samples = 0
    model.eval()

    answer = -1
    with torch.no_grad():
        for x, y in loader:

            scores = model(x)
            
            _, predictions = scores.max(1)
            _, true_value = y.to(device).max(1)
            num_correct += (predictions == true_value).sum()
            num_samples += predictions.size(0)

            if first_n_samples is not None and num_samples > first_n_samples:
                break
        
        answer = float(num_correct)/float(num_samples) if num_samples > 0 else 0

    model.train()
    return answer
