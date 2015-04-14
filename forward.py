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
from unidecode import unidecode
import gensim

warnings.filterwarnings("ignore")   

class CNN(object):

    """
    Class for CNN
    """

    def __init__(self,
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
        x = T.ftensor4('x')
        layer0_input = x.reshape((x.shape[0],1,x.shape[2],x.shape[3]))
        self.conv_layers = []
        layer1_inputs = []
        for i in xrange(len(filter_hs)):
            # filter_shape = filter_shapes[len(filter_hs)-1-i]
            # print filter_shape
            # pool_size = pool_sizes[len(filter_hs)-1-i]
            filter_shape = filter_shapes[i]
            pool_size = pool_sizes[i]
            c = 2*(len(filter_hs)-i)+1
            conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,image_shape=(batch_size, 1, img_h, img_w),
                                    filter_shape=filter_shape, params_loaded= [params_loaded[c-1],params_loaded[c]], name_model = "cnet_"+str(i), poolsize=pool_size, non_linear=conv_non_linear)
            layer1_input = conv_layer.output.flatten(2)
            self.conv_layers.append(conv_layer)
            layer1_inputs.append(layer1_input)
        self.layer1_input = T.concatenate(layer1_inputs,1)
        hidden_units[0] = feature_maps*len(filter_hs)
        self.classifier = MLPDropout(rng, input=layer1_input, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate, params = [params_loaded[0], params_loaded[1]])

    def predict(self):
        test_pred_layers = []
        x = T.ftensor4('x')
        test_layer0_input = x
        for i in range(len(self.conv_layers)):
            test_layer0_output = self.conv_layers[i].predict(test_layer0_input, 1)
            test_pred_layers.append(test_layer0_output.flatten(2))
        test_layer1_input = T.concatenate(test_pred_layers, 1)
        test_y_pred = self.classifier.predict_p(test_layer1_input)
        f = theano.function([x], test_y_pred)
        return f

def sen2mat(sen, w2v, max_l=900, k=300, filter_h=5):

    pad = filter_h - 1
    mat = np.zeros((max_l+2*pad,k),dtype=theano.config.floatX)
    i = 0
    for word in sen.split(" "):
        if word in w2v:
            mat[i + pad] = w2v[word]
        elif unidecode(word) in w2v:
            mat[i + pad] = w2v[unidecode(word)]
        else:
            mat[i + pad] = np.zeros(k)
    mat = mat.reshape((1,1,max_l+2*pad,k))
    return mat

def parse_filter_hs(filter_hs):
    return map(int, filter_hs.split(','))

if __name__=="__main__":

    # Arguments for the program:
    parser = argparse.ArgumentParser(description='Convnet')
    parser.add_argument('--load_weights', default='weights.p')
    parser.add_argument('--lang')
    parser.add_argument('w2v')
    args = parser.parse_args()
    print "loading model...",
    x = cPickle.load(open(args.load_weights,"rb"))
    # Create a CNN object with the params loaded
    model = CNN(x)
    print "model loaded!"

    print "loading w2v...",
    if args.lang == "fr":
        w2v = gensim.models.Word2Vec.load_word2vec_format(args.w2v +"fr.2gram.sem", binary=True)
    if args.lang == "en":
        w2v = gensim.models.Word2Vec.load_word2vec_format(args.w2v +"en.2gram.sem", binary=True)
    elif args.lang == "de":
        w2v = gensim.models.Word2Vec.load_word2vec_format(args.w2v +"de.2gram.sem", binary=True)
    elif args.lang == "it":
        w2v = gensim.models.Word2Vec.load_word2vec_format(args.w2v +"it.2gram.sem", binary=True)
    elif args.lang == "es":
        w2v = gensim.models.Word2Vec.load_word2vec_format(args.w2v +"es.2gram.sem", binary=True)
    elif args.lang == "zh":
        w2v = gensim.models.Word2Vec.load_word2vec_format(args.w2v +"zh.2gram.sem", binary=True)
    print "w2v loaded!"

    @Request.application
    def CNN_demo(request):
        result = ['<title>Write a sentence!</title>']
        result.append('''
            <form action="" method="post">
                <p>Sentence: <input type="text" name="sentence" size="20">
                <input type="submit" value="Let's compute that shit!">
            </form>
        ''')
        if request.method == 'POST':
            sen_test = escape(request.form['sentence'])
            prediction = model.predict()(x).get_value()
            result.append('<p>Positivity: %s</p>' %(prediction[0]))
            result.append('<p>Negativity: %s</p>' %(prediction[1]))
        return Response(''.join(result), mimetype='text/html')

    # Load the werkzeug serving
    port = np.random.randint(1, high=4000)
    run_simple('localhost', port, CNN_demo)

    # Create a POST request that returns the inference made by the CNN of the sentence posted
    # Process the sentence
