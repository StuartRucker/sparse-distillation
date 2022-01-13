from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(1, 4), max_features=100, input='filename') #make object of Count Vectorizer



X = vectorizer.fit(['data/imdb/train/pos/0_9.txt', 'data/imdb/train/pos/1_7.txt'])

    
print(X.transform(['data/imdb/train/pos/0_9.txt', 'data/imdb/train/pos/1_7.txt']).toarray())

print(X.get_feature_names_out())