"""
Sample code for
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf

Much of the code is modified from
- deeplearning.net (for ConvNet classes)
- https://github.com/mdenil/dropout (for dropout)
- https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
"""

import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import re
import warnings
import sys
import argparse
from conv_net_classes import *
from time import ctime
from werkzeug.utils import escape
from werkzeug.wrappers import Request, Response
from werkzeug.serving import run_simple

warnings.filterwarnings("ignore")   

class CNN(object):

    """
    Class for CNN
    """

    def __init__(self,
                 U,
                 params_loaded,
                 img_h = 908,
                 img_w = 300,
                 feature_maps = 100,
                 filter_hs = [3,4,5],
                 dropout_rate = [0.5],
                 batch_size = 1,
                 hidden_units = [100,2],
                 activations = ["ReLU"],
                 conv_non_linear = "relu"
                 ):

        rng = np.random.RandomState(3435)
        self.img_h = img_h
        self.img_w = img_w
        filter_w = img_w
        filter_shapes = []
        pool_sizes = []
        for filter_h in filter_hs:
            filter_shapes.append((feature_maps, 1, filter_h, filter_w))
            pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))

        #define model architecture
        index = T.lscalar()
        x = T.matrix('x')
        y = T.ivector('y')
        self.Words = theano.shared(value = np.asarray(U, dtype=theano.config.floatX), name = "Words")
        zero_vec_tensor = T.vector()
        zero_vec = np.zeros(img_w)
        set_zero = theano.function([zero_vec_tensor], updates=[(self.Words, T.set_subtensor(self.Words[0,:], zero_vec_tensor))], allow_input_downcast = True)
        self.layer0_input = self.Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],1,x.shape[1],self.Words.shape[1]))
        self.conv_layers = []
        layer1_inputs = []
        for i in xrange(len(filter_hs)):
            filter_shape = filter_shapes[len(filter_hs)-1-i]
            pool_size = pool_sizes[len(filter_hs)-1-i]
            c = 2*(len(filter_hs)-i)+1
            print c, c-1
            print params_loaded[c-1].shape, params_loaded[c].shape
            conv_layer = LeNetConvPoolLayer(rng, input=self.layer0_input,image_shape=(batch_size, 1, img_h, img_w),
                                    filter_shape=filter_shape, params_loaded= [params_loaded[c-1],params_loaded[c]], name_model = "cnet_"+str(i), poolsize=pool_size, non_linear=conv_non_linear)
            layer1_input = conv_layer.output.flatten(2)
            self.conv_layers.append(conv_layer)
            layer1_inputs.append(layer1_input)
        self.layer1_input = T.concatenate(layer1_inputs,1)
        hidden_units[0] = feature_maps*len(filter_hs)
        self.classifier = MLPDropout(rng, input=self.layer1_input, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate)

    def predict(self):

        x = T.matrix('x')
        test_pred_layers = []
        test_layer0_input = self.Words[T.cast(x.flatten(),dtype="int32")].reshape((1,1,self.img_h,self.Words.shape[1]))
        for conv_layer in self.conv_layers:
            test_layer0_output = conv_layer.predict(test_layer0_input, 1)
            test_pred_layers.append(test_layer0_output.flatten(2))
        test_layer1_input = T.concatenate(test_pred_layers, 1)
        test_y_pred = self.classifier.predict_p(test_layer1_input)
        f = theano.function([x], test_y_pred)
        return f

def get_idx_from_sent(sent, word_idx_map, max_l=900, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    print x
    return x

def make_idx_data_cv(revs, word_idx_map, cv, max_l=51, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)
        sent.append(rev["y"])
        if rev["split"]==cv:
            test.append(sent)
        else:
            train.append(sent)
    train = np.array(train)
    test = np.array(test)
    return [train, test]

def parse_filter_hs(filter_hs):
    return map(int, filter_hs.split(','))

if __name__=="__main__":

    # Arguments for the program:
    parser = argparse.ArgumentParser(description='Convnet')
    parser.add_argument('--load_weights', default='weights.p')
    parser.add_argument('--load_vocab', default='corpus.p')
    args = parser.parse_args()
    print "loading model...",
    c = cPickle.load(open(args.load_vocab,"rb"))
    revs, W, W2, word_idx_map, vocab = c[0], c[1], c[2], c[3], c[4]
    x = cPickle.load(open(args.load_weights,"rb"))
    # Create a CNN object with the params loaded
    model = CNN(W, x)
    print "model loaded!"

    @Request.application
    def CNN_demo(request):
        result = ['<title>Write a sentence!</title>']
        if request.method == 'POST':
            sen_test = escape(request.form['sentence'])
            sen_test = get_idx_from_sent(str(sen_test), word_idx_map, max_l=900, k=300, filter_h=5)
            x = np.array(sen_test, dtype=theano.config.floatX).reshape(1,len(sen_test))
            prediction = str(model.predict()(x))
            result.append('<h1>%s</h1>' %(prediction))
        result.append('''
            <form action="" method="post">
                <p>Sentence: <input type="text" name="sentence" size="20">
                <input type="submit" value="Let's compute that shit!">
            </form>
        ''')
        return Response(''.join(result), mimetype='text/html')

    # Load the werkzeug serving
    run_simple('localhost', 4000, CNN_demo)

    # Create a POST request that returns the inference made by the CNN of the sentence posted
    # Process the sentence
