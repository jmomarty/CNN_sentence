# -*- coding: utf-8 -*-

import numpy as np
import cPickle
import pandas as pd
import argparse
import gensim
import codecs
from unidecode import unidecode


def cd(k, lg, txt, nw, s):

    datum = {"y": k,
             "language": lg,
             "text": txt,
             "num_words": nw,
             "split": s}

    return datum


def create_dict(d, r, v, s, cv):

    tg = {}
    c = 0
    with codecs.open(d, "rb", encoding="utf-8") as f:
        for line in f:
            sen_array = line.split()
            lang = sen_array[0]
            target = sen_array[1]
            if target not in tg:
                tg[target] = c
                c += 1
            words = set(sen_array)
            if lang in v:
                for word in words:
                    v[lang][unicode(word)] = 1
            else:
                v[lang] = {}
                for word in words:
                    v[lang][unicode(word)] = 1
            if cv:
                datum = cd(tg[target], lang, unicode(" ".join(sen_array[2:])), len(sen_array[2:]), np.random.randint(0,s))
            else:
                datum = cd(tg[target], lang, unicode(" ".join(sen_array[2:])), len(sen_array[2:]), s)
            r.append(datum)

    return r, v


def build_data(splits, s=0, cv=None):

    revs = []
    vocab = {}

    if cv:  # cross validation
        revs, vocab = create_dict(splits[0], revs, vocab, s, cv)

    else:  # train/test split
        k = 0
        for split in splits:
            revs, vocab = create_dict(split, revs, vocab, k, cv)
            k += 1

    return revs, vocab


def get_w(wv_dict, k=300):

    """
    Get word matrix. W[i] is the vector for word indexed by i
    """

    mapping = {}
    vocab_size = 0
    for x in wv_dict:
        vocab_size += len(wv_dict[x])
    w = np.zeros(shape=(vocab_size+1, k))
    w[0] = np.zeros(k)
    i = 1
    for lg in wv_dict:
        mapping[lg] = {}
        for word in wv_dict[lg]:
            w[i] = wv_dict[lg][word]
            mapping[lg][word] = i
            i += 1

    return w, mapping


def load_bin_vec(vocab, w2v):

    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """

    wv = {}
    for lg in w2v:
        print lg
        lgm = gensim.models.Word2Vec.load_word2vec_format(lg, binary=True)
        wv[lg] = {}
        print lgm.vocab
        for word in lgm.vocab:
            if word in vocab["fr"]:
                wv[lg][word] = lgm[word]
            elif unidecode(word) in vocab["fr"]:
                wv[lg][unidecode(word)] = lgm[word]

    return wv


def add_unknown_words(vocab, wv, min_df=1, k=300):

    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """

    for lg in vocab:
        for word in vocab[lg]:
            if word not in wv[lg]:
                if vocab[lg][word] >= min_df:
                    wv[lg][word] = np.random.uniform(-0.25, 0.25, k)

    return wv


if __name__ == "__main__":

    # Arguments for the program:
    parser = argparse.ArgumentParser(description='Data Processing')
    parser.add_argument('mode', help='cv/dev')
    parser.add_argument('--w2v', nargs='*')
    parser.add_argument('--w2v_size', default=300)
    parser.add_argument('--train_files', nargs='*')
    parser.add_argument('--dev_files', nargs='*')
    parser.add_argument('--test_files', nargs='*')
    parser.add_argument('--output', default='data.p')
    parser.add_argument('--addrandomvec', default=0)

    args = parser.parse_args()

    w2v = args.w2v
    w2v_size = int(args.w2v_size)

    train_folder = args.train_files
    dev_folder = args.dev_files
    test_folder = args.test_files

    print "loading data...",

    if args.mode == "dev":
        rvs, vcb = build_data([train_folder, dev_folder, test_folder])
    else:
        rvs, vcb = build_data(train_folder, s=int(args.mode), cv=True)

    pd_data_num_words = pd.DataFrame(rvs)["num_words"]
    max_l = np.max(pd_data_num_words)
    mean_l = np.mean(pd_data_num_words)
    class_dist = pd.DataFrame(rvs)["y"].values
    class_dist, _ = np.histogram(class_dist, bins=50)

    print "data loaded!"
    print "number of sentences: " + str(len(rvs))
    l = 0
    for lg in vcb:
        l += len(vcb[lg])
    print "vocab size: " + str(l)
    print "max sentence length: " + str(max_l)
    print "average sentence length: " + str(mean_l)
    print "class distribution: " + str(class_dist)

    print "loading word2vec vectors...",
    wv = load_bin_vec(vcb, w2v)
    print "word2vec loaded!"

    l = 0
    for lg in wv:
        l += len(wv[lg])
    print "num words already in word2vec: " + str(l)

    if args.addrandomvec == 1:
        wv = add_unknown_words(vcb, wv, k=w2v_size)
    else:
        W, mapping = get_w(wv, k=w2v_size)

    cPickle.dump([rvs, W, mapping], open(args.output, "wb"))
    print "dataset created!"