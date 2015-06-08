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
                 weights,
                 s_h=50,
                 s_w=300,
                 reshape=30,
                 feature_maps=100,
                 filter_hs=None,
                 dropout_rate=None,
                 batch_size=1,
                 hidden_units=None,
                 activations=None,
                 conv_non_linear="relu",
                 ):


        if not filter_hs:
            filter_hs = [3, 4, 5]
        if not hidden_units:
            hidden_units = [100, 3]
        if not dropout_rate:
            dropout_rate = [0.5]
        if not activations:
            activations = ["ReLU"]

        rng = np.random.RandomState(3435)
        self.img_h = s_h
        self.img_w = s_w
        filter_w = s_w
        filter_shapes = []
        pool_sizes = []
        for filter_h in filter_hs:
            filter_shapes.append((feature_maps, 1, filter_h, filter_w))
            pool_sizes.append((s_h-filter_h+1, s_w-filter_w+1))

        # define model architecture
        x = T.ftensor4('x')
        layer0_input = x.reshape((x.shape[0], 1, x.shape[2], x.shape[3]))
        self.layer1 = HiddenLayer(rng, layer0_input, n_in=s_w, n_out=reshape, activation=ReLU,
                                  w=weights[8], b=weights[9], use_bias=True)
        layer1_input = self.layer1.output
        self.conv_layers = []
        layer2_inputs = []
        for i in xrange(len(filter_hs)):
            filter_shape = filter_shapes[i]
            pool_size = pool_sizes[i]
            c = 2*(len(filter_hs)-i)+1
            conv_layer = LeNetConvPoolLayer(rng, ipt=layer1_input,image_shape=(batch_size, 1, s_h, s_w),
                                            filter_shape=filter_shape, params_loaded= [weights[c-1],weights[c]], name_model = "cnet_"+str(i), poolsize=pool_size, non_linear=conv_non_linear)
            layer2_input = conv_layer.output.flatten(2)
            self.conv_layers.append(conv_layer)
            layer2_inputs.append(layer2_input)
        layer2_input = T.concatenate(layer2_inputs,1)
        hidden_units[0] = feature_maps*len(filter_hs)
        self.classifier = MLPDropout(rng, ipt=layer2_input, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate, params = [weights[0], weights[1]])

    def predict(self):
        test_pred_layers = []
        x = T.ftensor4('x')
        test_layer0_input = x
        test_layer1_input = self.layer1.predict(test_layer0_input)
        for i in range(len(self.conv_layers)):
            test_layer2_input = self.conv_layers[i].predict(test_layer1_input, 1)
            test_pred_layers.append(test_layer2_input.flatten(2))
        test_layer3_input = T.concatenate(test_pred_layers, 1)
        test_y_pred = self.classifier.predict_p(test_layer3_input)
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
        i+=1
    mat = mat.reshape((1,1,max_l+2*pad,k))
    return mat

def parse_filter_hs(filter_hs):
    return map(int, filter_hs.split(','))

if __name__=="__main__":

    # Arguments for the program:
    parser = argparse.ArgumentParser(description='Convnet')
    parser.add_argument('lang')
    parser.add_argument('--load_weights', default='ciao300')

    args = parser.parse_args()
    print "loading model...",
    x = cPickle.load(open(args.load_weights,"rb"))
    # Create a CNN object with the params loaded
    model = CNN(x)
    print "model loaded!"

    print "loading w2v...",
    w2v = gensim.models.Word2Vec.load_word2vec_format("F:\\mikolov\\"+args.lang+".2gram.sem", binary=True)
    print "w2v loaded!\n"

    @Request.application
    def CNN_demo(request):
        result = ['<title>%s</title>' %(str(args.lang))]
        result.append('''
            <form action="" method="post">
                <p>Sentence: <input type="text" name="sentence" size="20">
                <input type="submit" value="Let's compute that shit!">
            </form>
        ''')
        if request.method == 'POST':
            sen_test = escape(request.form['sentence']).lower()
            x = sen2mat(sen_test, w2v)
            prediction = model.predict()(x)
            result.append('<p>Polarity: %s</p>' %(prediction[0,0]))
            pgrams = {}
            words = sen_test.split(" ")
            for i in range(1,5):
                for gram in zip(*[words[k:] for k in range(i)]):
                    gram_mat = sen2mat(" ".join(gram), w2v)
                    pgrams[gram] = model.predict()(gram_mat)[0,0]
            import operator
            sorted_p = sorted(pgrams.items(), key=operator.itemgetter(1), reverse=True)
            result.append('<p>Polarity of N-grams:</p>')
            for x in sorted_p:
                result.append('<p>%s</p>' %str(x))
        return Response(''.join(result), mimetype='text/html')

    # Load the werkzeug serving
    port = np.random.randint(1, high=4000)
    run_simple('localhost', port, CNN_demo)

