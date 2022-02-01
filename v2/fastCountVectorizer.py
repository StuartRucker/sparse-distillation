from util import extract_ngrams

class FastCountVectorizer:
    def __init__(self, filename):
        self.filename = filename
        self.word_to_index = {}
        with open(filename, 'r') as f:
            for i, line in enumerate(f.readlines()):
                self.word_to_index[line.rstrip('\n')] = i+1

    def tokenize(self, text):
        ngrams = extract_ngrams(text)
        return [self.word_to_index[ngram] for ngram in ngrams if ngram in self.word_to_index]  

    def size(self):
        return len(self.word_to_index)
