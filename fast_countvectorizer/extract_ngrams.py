# import pymongo
# from bson.objectid import ObjectId
# from tqdm import tqdm
# myclient = pymongo.MongoClient("mongodb://localhost:27017/")
# mydb = myclient["mydatabase"]
# mycol = mydb["wordcounts"]
import re

# review_filepath =  '/Users/stuartrucker/nlp/yoon/data/reviews/reviews.txt'

def is_word(s):
    return not s.isnumeric()
def get_ngrams(s):
    all_words = re.findall( r'(?u)\b\w\w+\b', s.lower())
    ngrams = []
    for i in range(len(all_words)):
        for j in range(i+1, min(i+5, len(all_words))):
            ngrams.append(' '.join(all_words[i:j]))
    return ngrams

def pad_word(s):
    return str.encode("_"*(24-len(s)) + s if len(s) < 24 else s)
def package_ngrams(ngrams):
    return [{ '_id' : ngram, '$inc' : { 'val' : 1 } } for ngram in ngrams]

with open('output.txt', 'w') as fp_out:
    with open('data/reviews/reviews.txt') as fp:
        for i in tqdm(range(10000)):
            text = fp.readline()
            ngrams = get_ngrams(text)
            for ngram in ngrams:
                print(ngram, file = fp_out)
            # mycol.update_one({ 'ngram' : ngram}, {'$inc' : { 'val' : 1 } } , upsert=True)
        # mycol.insert_many([{ 'ngram' : ngram, 'val':1} for ngram in ngrams])

