import torch
import glob
#import DAN module from model.py
from model import DAN
#import FastCountVectorizer
from fastCountVectorizer import FastCountVectorizer
#import IMDBDataset from datasets/imdbDataset.py
from datasets.imdbDataset import ImdbDataset

#import torch Dataset
from torch.utils.data import Dataset, DataLoader

import random

tokenizer = FastCountVectorizer('../data/fast_countvectorizer/top_million_ngrams.txt')

config = {
    'embed_dimension': 1000,
    'intermediate_dimension': 1000
}

model = DAN(
    num_embeddings=tokenizer.size()+1,
    embed_dimension=config['embed_dimension'],
    intermediate_dimension=config['intermediate_dimension']
)

#load the model from the checkpoint
old_state_dict = torch.load('model_best.pth.tar', map_location='cpu')['state_dict']
#rename all the keys in the state_dict to not contain "module."
new_state_dict = {}
for key in old_state_dict:
    new_state_dict[key.replace('module.', '')] = old_state_dict[key]
model.load_state_dict(new_state_dict)



#load the dataset
test_files = glob.glob("../data/imdb/test/*/*.txt")
#randomly sample 100 files from the test set
test_files = random.sample(test_files, 2000)
dataset = ImdbDataset(test_files, tokenizer)

#get the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)



model.eval()
num_correct, total = 0,0

#add zeros until a has length 1000
def pad_with_zeros(a):
    return a + [0] * (1000 - len(a))
def prepare_text(s):
    return torch.LongTensor([pad_with_zeros(tokenizer.tokenize(s))])

#iterate through each batch
for batch_idx, (data, target) in enumerate(dataloader): 
    #load the batch
    #data, target = data.to(torch.device('cuda')), target.to(torch.device('cuda'))
    #make predictions
    output = model(data)
    #check if the prediction is correct
    _, predicted = torch.max(output.data, 1)

    _, target_value = torch.max(target, 1)
    #increment the number of correct predictions and the total number of predictions
    num_correct += (predicted == target_value).sum().item()
    total += target.size(0)

#print the accuracy
print("Accuracy: %.2f %%" % (100 * num_correct / total))
    


