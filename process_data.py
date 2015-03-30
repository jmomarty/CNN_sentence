import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd
import argparse

def build_data_cv(data, cv=10, clean=True):

    """
    Loads data and split into cv folds.
    """

    revs = []
    vocab = defaultdict(float)

    for k in sorted(data):
        with open(data[k], "rb") as f:
            for line in f:
                rev = []
                rev.append(line.strip())
                if clean:
                    orig_rev = clean_str(" ".join(rev))
                else:
                    orig_rev = " ".join(rev).lower()
                words = set(orig_rev.split())
                for word in words:
                    vocab[word] += 1
                datum  = {"y": k,
                          "text": orig_rev,
                          "num_words": len(orig_rev.split()),
                          "split": np.random.randint(0,cv)}
                revs.append(datum)

    return revs, vocab

def build_data(train, test, clean=True):

    revs = []
    vocab = defaultdict(float)
    c = 0

    for x in train:
        with open(x, "rb") as f:
            for line in f:
                if clean:
                    line = clean_str(line)
                else:
                    line = line.lower()
                words = line.split()
                for word in set(words):
                    vocab[word] += 1
                datum  = {"y": c,
                          "text": line,
                          "num_words": len(words),
                          "split": 0}
                revs.append(datum)
        c += 1

    c = 0

    for x in test:
        with open(x, "rb") as f:
            for line in f:
                if clean:
                    line = clean_str(line)
                else:
                    line = line.lower()
                words = line.split()
                for word in set(words):
                    vocab[word] += 1
                datum  = {"y": c,
                          "text": line,
                          "num_words": len(words),
                          "split": 1}
                revs.append(datum)
        c += 1

    return revs, vocab

def build_data_tdt(train, dev, test, clean=True):

    revs = []
    vocab = defaultdict(float)
    c = 0

    for x in train:
        with open(x, "rb") as f:
            for line in f:
                if clean:
                    line = clean_str(line)
                else:
                    line = line.lower()
                words = line.split()
                for word in set(words):
                    vocab[word] += 1
                datum  = {"y": c,
                          "text": line,
                          "num_words": len(words),
                          "split": 0}
                revs.append(datum)
        c += 1

    c = 0

    for x in dev:
        with open(x, "rb") as f:
            for line in f:
                if clean:
                    line = clean_str(line)
                else:
                    line = line.lower()
                words = line.split()
                for word in set(words):
                    vocab[word] += 1
                datum  = {"y": c,
                          "text": line,
                          "num_words": len(words),
                          "split": 1}
                revs.append(datum)
        c += 1
    c = 0

    for x in test:
        with open(x, "rb") as f:
            for line in f:
                if clean:
                    line = clean_str(line)
                else:
                    line = line.lower()
                words = line.split()
                for word in set(words):
                    vocab[word] += 1
                datum  = {"y": c,
                          "text": line,
                          "num_words": len(words),
                          "split": 2}
                revs.append(datum)
        c += 1

    return revs, vocab

def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k))            
    W[0] = np.zeros(k)
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

if __name__=="__main__":

    # Arguments for the program:
    parser = argparse.ArgumentParser(description='Data Processing')
    parser.add_argument('mode', help='cv/dev')
    parser.add_argument('word_vectors', help='w2v_file')
    parser.add_argument('--train_files', nargs = '*')
    parser.add_argument('--dev_files', nargs = '*')
    parser.add_argument('--test_files', nargs = '*')
    parser.add_argument('--clean', default=True)
    parser.add_argument('--output', default='data.p')
    args = parser.parse_args()

    w2v_file = args.word_vectors

    train_folder = args.train_files
    dev_folder = args.dev_files
    test_folder = args.test_files

    print "loading data...",
    if args.mode != "dev":
        revs, vocab = build_data_cv(train_folder, int(args.mode), args.clean)
    else:
        revs, vocab = build_data(train_folder, dev_folder, test_folder, args.clean)

    pd_data_num_words = pd.DataFrame(revs)["num_words"]
    max_l = np.max(pd_data_num_words)
    mean_l = np.mean(pd_data_num_words)
    class_dist = pd.DataFrame(revs)["y"].values
    class_dist, _ = np.histogram(class_dist, bins = len(train_folder))
    print "data loaded!"
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    print "average sentence length: " + str(mean_l)
    print "class distribution: " + str(class_dist)

    print "loading word2vec vectors...",
    w2v = load_bin_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)

    cPickle.dump([revs, W, W2, word_idx_map, vocab], open(args.output, "wb"))
    print "dataset created!"
    
