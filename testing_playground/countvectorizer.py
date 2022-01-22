from sklearn.feature_extraction.text import CountVectorizer
import pickle
from tqdm import tqdm

vectorizer = CountVectorizer(ngram_range=(1, 4), max_features=1000000) #make object of Count Vectorizer




with open('data/reviews/reviews.txt') as fp:
    vectorizer.fit((fp.readline() for i in tqdm(range(75000000))))


filename = 'vectorizer.sav'
pickle.dump(vectorizer, open(filename, 'wb')) 

