from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
device = 0 if torch.cuda.is_available() else -1

model_name = "siebert/sentiment-roberta-large-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_len=512)
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, device=device)


def longstr():
    s = ''
    for i in range(500):
        s += "I hate this! ⭻⬚⣷⪃❰⥑⢔✵⩮∨◞⋠⨤⠚℻◌ "
    return s

print(classifier(["""I've seen quite a few negative reviews, and I can somewhat understand in the case that one has no patience with setting up new technologies. If you want Lifx and Alexa to get on well, it'll take a bit of time and a little effort (read a little, think about it a little, etc.). One way to make things very simple is to use IFTTT (If This, Then That) which invokes simple conditional commands (unfortunately at this point complex conditionals, e.g. If x and y, then z, are not an option, as far as I can tell). So, for example, you can set lights to a particular color (say, orange) and a particular brightness (say, 30%), then include those parameters in what Lifx calls a scene (this is all done in the Lifx app). Then you can use IFTTT to execute that scene with a simple command like "Alexa, trigger waking up" or whatever you wish. The point here is that the poor reviews are not reflective of limitations imposed by Lifx lights or Alexa (this is not to say there aren't limits to both and to their interaction, but that the poor reviews do not come close to reflecting those limitations). I am now able to control lights in every room of the house with voice commands to Echo. A couple of times it hasn't worked. And both times it was because I had left something out of the command (whether in Alexa, IFTTT, or Lifx). As for Echo not understanding "Lif-eX", this was the case until a recent update with Echo/Alexa a few weeks ago. It was frustrating, since people claimed to be able to do it. I could not. Then Amazon sent out an update to Echo/Alexa and it worked great. It works best when "Lifx" is articulated clearly, i.e. "Life Ex". In the early stages of home automation, things can, as we''d expect, be a bit tricky/buggy, and I don't mean to suggest that everything works as we would ideally like, but it's pretty cool. I can walk into my house and with a couple of voice commands turn on the stereo and TV (they're hooked together for movies), set the lights to a particular brightness and color, and change the temperature. This is better than most royalty had it in days past. So if you're looking to automate your house, this is a pretty cool setup. Yes, it'll take a little time and effort, but so will any other setup. I'm very pleased. And, no, I have no affiliation or association with either Lifx or Amazon. One thing I would like to see implemented is for Echo to enable direct scene commands for Lifx."""], padding=True, truncation=True))