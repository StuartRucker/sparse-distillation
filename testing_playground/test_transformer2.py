from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from scipy.special import softmax

import sys
import torch




device = "cuda:0" if torch.cuda.is_available() else "cpu"

model_name = "siebert/sentiment-roberta-large-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)#, padding=True, max_length=512, truncation=True)
#classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, device=device)

def classifier(text):
    tokenized_text = tokenizer(text, max_length=512, truncation=True, padding=True, return_tensors="pt").to(device)
    return softmax(model(**tokenized_text).logits.detach().numpy(), axis=0)[:,1]

print(classifier( ["I love you "*500, "I hate you"]))