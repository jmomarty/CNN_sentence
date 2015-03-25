import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd
import argparse

# Arguments for the program:
parser = argparse.ArgumentParser(description='Data Stats')
parser.add_argument('--input', default='mr.p')
args = parser.parse_args()

w2v_file = args.word_vectors

train_folder = args.train_files
test_folder = args.test_files

print "loading data...",
x = cPickle.load(open(args.input,"rb"))
revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]

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
