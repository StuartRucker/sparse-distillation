
import os
import random
import tqdm

def main(args):
    
    saved_ngrams = set()

    with open(args['input_file'], 'r') as f_in:
        with open(args['output_file'], 'w') as f_out:
            #read the file f line by line without saving it all to memory
            for line in tqdm.tqdm(f_in):
                words = line.split()

                f_out.write(line)
                if len(words) > 2: #only mask short ngrams
                    continue
                for i, masked_word in enumerate(words):
                    new_words_list = words[:i] + ['[MASK]'] + words[i+1:]
                    new_ngram = ' '.join(new_words_list)
                    if new_ngram not in saved_ngrams:
                        f_out.write(new_ngram + '\n')
                        saved_ngrams.add(new_ngram)


if __name__ == '__main__':
    args = {
        'input_file': "../../data/fast_countvectorizer/top_million_ngrams.txt",
        'output_file': "../../data/fast_countvectorizer/top_ngrams_masked.txt"
    }

    args['input_file'] = os.path.join(os.path.dirname(__file__), args['input_file'])
    args['output_file'] = os.path.join(os.path.dirname(__file__), args['output_file'])


    main(args)