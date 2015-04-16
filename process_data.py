# -*- coding: utf-8 -*-
import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd
import argparse
import gensim
import codecs
from unidecode import unidecode

def build_data_cv(data, cv=10, clean=True):

    """
    Loads data and split into cv folds.
    """

    revs = []
    vocab_en = defaultdict(float)
    vocab_fr = defaultdict(float)

    for k in range(len(data)):
        with codecs.open(data[k], "rb", encoding="utf-8") as f:
            for line in f:
                sen_array = line.split()
                lang = sen_array[0]
                words = set(sen_array)
                if lang == "en":
                    for word in words:
                        vocab_en[unicode(word)] += 1
                if lang == "fr":
                    for word in words:
                        vocab_fr[unicode(word)] += 1
                datum  = {"y": k,
                          "language":lang,
                          "text": unicode(" ".join(sen_array[1:])),
                          "num_words": len(sen_array[1:]),
                          "split": np.random.randint(0,cv)}
                revs.append(datum)

    return revs, vocab_fr, vocab_en

def build_data(train, test, clean=True):

    revs = []
    vocab = defaultdict(float)
    c = 0

    for x in train:
        with codecs.open(x, "rb", encoding="utf-8") as f:
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
        with codecs.open(x, "rb", encoding="utf-8") as f:
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
        with codecs.open(x, "rb", encoding="utf-8") as f:
            for line in f:
                words = line.split()
                for word in set(words):
                    vocab[word] += 1
                datum  = {"y": c,
                          "text": unicode(line),
                          "num_words": len(words),
                          "split": 0}
                revs.append(datum)
        c += 1

    c = 0

    for x in dev:
        with codecs.open(x, "rb", encoding="utf-8") as f:
            for line in f:
                words = line.split()
                for word in set(words):
                    vocab[word] += 1
                datum  = {"y": c,
                          "text": unicode(line),
                          "num_words": len(words),
                          "split": 1}
                revs.append(datum)
        c += 1
    c = 0

    for x in test:
        with codecs.open(x, "rb", encoding="utf-8") as f:
            for line in f:
                words = line.split()
                for word in set(words):
                    vocab[word] += 1
                datum  = {"y": c,
                          "text": unicode(line),
                          "num_words": len(words),
                          "split": 2}
                revs.append(datum)
        c += 1

    return revs, vocab

def get_W(w2v_fr, w2v_en, k=300):

    """
    Get word matrix. W[i] is the vector for word indexed by i
    """

    vocab_fr_size = len(w2v_fr)
    vocab_en_size = len(w2v_en)
    W_fr = np.zeros(shape=(vocab_fr_size+1, k))
    W_fr[0] = np.zeros(k)
    W_en = np.zeros(shape=(vocab_en_size+1, k))
    W_en[0] = np.zeros(k)
    i = 1
    for word in w2v_fr:
        W_fr[i] = w2v_fr[word]
        i += 1
    for word in w2v_en:
        W_en[i] = w2v_en[word]
        i += 1
    return W_fr, W_en, word_idx_map_fr, word_idx_map_en

def load_bin_vec(vocab_fr, vocab_en, fr, en):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    wv_fr = {}
    wv_en = {}
    fr = gensim.models.Word2Vec.load_word2vec_format(fr, binary=True)
    en = gensim.models.Word2Vec.load_word2vec_format(en, binary=True)
    for word in fr.vocab:
        if word in vocab_fr:
            wv_fr[word] = fr[word]
        if unidecode(word) in vocab_fr:
            wv_fr[unidecode(word)] = fr[word]
    for word in en.vocab:
        if word in vocab_en:
            wv_en[word] = en[word]
        if unidecode(word) in vocab_en:
            wv_en[unidecode(word)] = en[word]

    return wv_fr, wv_en

def add_unknown_words(vocab_fr, vocab_en, wv_fr, wv_en, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """

    for word in vocab_fr:
        if word not in wv_fr:
            if vocab_fr[word] >= min_df:
                wv_fr[word] = np.random.uniform(-0.25,0.25,k)
    for word in vocab_en:
        if word not in wv_en:
            if vocab_en[word] >= min_df:
                wv_en[word] = np.random.uniform(-0.25,0.25,k)

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
    parser.add_argument('w2v_fr')
    parser.add_argument('w2v_en')
    parser.add_argument('--train_files', nargs = '*')
    parser.add_argument('--dev_files', nargs = '*')
    parser.add_argument('--test_files', nargs = '*')
    parser.add_argument('--clean', default=False)
    parser.add_argument('--output', default='data.p')
    parser.add_argument('--w2v_size', default=300)
    args = parser.parse_args()

    w2v_fr = args.w2v_fr
    w2v_en = args.w2v_en
    w2v_size = int(args.w2v_size)

    train_folder = args.train_files
    dev_folder = args.dev_files
    test_folder = args.test_files

    print "loading data...",

    if args.mode != "dev":
        revs, vocab_fr, vocab_en = build_data_cv(train_folder, int(args.mode), args.clean)
    else:
        revs, vocab = build_data_tdt(train_folder, dev_folder, test_folder, args.clean)

    pd_data_num_words = pd.DataFrame(revs)["num_words"]
    max_l = np.max(pd_data_num_words)
    mean_l = np.mean(pd_data_num_words)
    class_dist = pd.DataFrame(revs)["y"].values
    class_dist, _ = np.histogram(class_dist, bins = len(train_folder))
    print "data loaded!"
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab_fr)+len(vocab_en))
    print "max sentence length: " + str(max_l)
    print "average sentence length: " + str(mean_l)
    print "class distribution: " + str(class_dist)

    print "loading word2vec vectors...",
    w2v = load_bin_vec(vocab_fr, vocab_en, w2v_fr, w2v_en)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    add_unknown_words(vocab_fr, vocab_en, w2v_fr, w2v_en, k = w2v_size)
    W_fr, W_en, word_idx_map_fr, word_idx_map_en = get_W(w2v_fr, w2v_en, k = w2v_size)

    cPickle.dump([revs, W_fr, W_en, vocab_fr, vocab_en], open(args.output, "wb"))
    print "dataset created!"