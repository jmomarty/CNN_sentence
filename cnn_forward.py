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
from collections import OrderedDict
import theano.tensor as T
import warnings
import argparse
import random
from conv_net_classes import *


warnings.filterwarnings("ignore")   


def train_conv_net(dst,
                   wv,
                   revs,
                   model_name,
                   params_loaded=None,
                   img_w=300,
                   filter_hs=[3,4,5],
                   hidden_units=[100,3],
                   dropout_rate=[0.5],
                   shuffle_batch=True,
                   n_epochs=25, 
                   batch_size=50, 
                   lr_decay = 0.95,
                   conv_non_linear="relu",
                   use_valid_set=True,
                   activations=["ReLU"],
                   sqr_norm_lim=9,
                   non_static=True):

    """
    Train a simple conv net
    img_h = sentence length (padded where necessary)
    img_w = word vector length (300 for word2vec)
    filter_hs = filter window sizes    
    hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
    sqr_norm_lim = s^2 in the paper
    lr_decay = adadelta decay parameter
    """

    rng = np.random.RandomState(3435)
    img_h = dst[0].shape[1]-1
    filter_w = img_w    
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, 30))
        pool_sizes.append((int(img_h-filter_h+1), int(img_w-filter_w+1)))
    parameters = [("image shape",img_h,img_w),("filter shape",filter_shapes), ("hidden_units",hidden_units),
                  ("dropout", dropout_rate), ("batch_size",batch_size),("non_static", non_static),
                    ("learn_decay",lr_decay), ("conv_non_linear", conv_non_linear), ("non_static", non_static)
                    ,("sqr_norm_lim",sqr_norm_lim),("shuffle_batch",shuffle_batch)]
    print parameters

    #define model architecture

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    Words = theano.shared(value=np.asarray(wv, dtype=theano.config.floatX), name="Words")
    layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0], 1, x.shape[1], 300))
    layer1 = HiddenLayer(rng, layer0_input, n_in=300, n_out=30, activation=ReLU, w=params_loaded[-2].get_value(), b=params_loaded[-1].get_value(), use_bias=True)
    layer1_input = layer1.output

    conv_layers = []
    layer2_inputs = []
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        if params_loaded == None:
            conv_layer = LeNetConvPoolLayer(rng, ipt=layer1_input,image_shape=(batch_size, 1, img_h, 30),
                                    filter_shape=filter_shape, params_loaded= params_loaded, name_model = "cnet_"+str(i), poolsize=pool_size, non_linear=conv_non_linear)
        else:
            c = 2*(len(filter_hs)-i)+1
            conv_layer = LeNetConvPoolLayer(rng, ipt=layer1_input,image_shape=(batch_size, 1, img_h, 30),
                                    filter_shape=filter_shape, params_loaded= [params_loaded[c-1],params_loaded[c]], name_model = "cnet_"+str(i), poolsize=pool_size, non_linear=conv_non_linear)
        layer2_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer2_inputs.append(layer2_input)
    layer2_input = T.concatenate(layer2_inputs,1)
    hidden_units[0] = feature_maps*len(filter_hs)
    print hidden_units
    if params_loaded == None:
        classifier = MLPDropout(rng, ipt=layer2_input, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate)
    else:
        classifier = MLPDropout(rng, ipt=layer2_input, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate, params = [params_loaded[0], params_loaded[1]])

    #define parameters of the model and update functions using adadelta
    params = classifier.params
    for conv_layer in conv_layers:
        params += conv_layer.params
    params += layer1.params
    cost = classifier.negative_log_likelihood(y) 
    dropout_cost = classifier.dropout_negative_log_likelihood(y)           
    grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)

    #shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate
    #extra data (at random)
    np.random.seed(3435)
    if datasets[0].shape[0] % batch_size > 0:
        extra_data_num = batch_size - datasets[0].shape[0] % batch_size
        train_set = np.random.permutation(datasets[0])
        extra_data = train_set[:extra_data_num]
        new_data=np.append(datasets[0],extra_data,axis=0)
    else:
        new_data = datasets[0]
    new_data = np.random.permutation(new_data)
    n_batches = new_data.shape[0]/batch_size
    n_train_batches = int(np.round(n_batches*0.9))
    print "n_train_batches : " + str(n_train_batches)
    if len(datasets)==3:
        print "using train/dev/test.."
        use_valid_set=True
        train_set = new_data
        val_set = datasets[1]
        train_set_x, train_set_y = shared_dataset((train_set[:,:img_h],train_set[:,-1]))
        val_set_x, val_set_y = shared_dataset((val_set[:,:img_h],val_set[:,-1]))
        test_set_x = datasets[2][:,:img_h]
        test_set_y = np.asarray(datasets[2][:,-1],"int32")
        n_val_batches = int(val_set.shape[0] / batch_size)
        print "n_val_batches : " + str(n_val_batches)
        val_errors = theano.function([index], classifier.errors(y),
            givens={
                  x: val_set_x[index * batch_size: (index + 1) * batch_size],
                  y: val_set_y[index * batch_size: (index + 1) * batch_size]})
    else:
        test_set_x = datasets[1][:,:img_h]
        test_set_y = np.asarray(datasets[1][:,-1],"int32")
        if use_valid_set:
            train_set = new_data[:n_train_batches*batch_size,:]
            val_set = new_data[n_train_batches*batch_size:,:]
            train_set_x, train_set_y = shared_dataset((train_set[:,:img_h],train_set[:,-1]))
            val_set_x, val_set_y = shared_dataset((val_set[:,:img_h],val_set[:,-1]))
            n_val_batches = n_batches - n_train_batches
            print n_val_batches
            val_errors = theano.function([index], classifier.errors(y),
                 givens={
                    x: val_set_x[index * batch_size: (index + 1) * batch_size],
                    y: val_set_y[index * batch_size: (index + 1) * batch_size]})
        else:
            train_set = new_data[:,:]
            train_set_x, train_set_y = shared_dataset((train_set[:,:img_h],train_set[:,-1]))

    train_model = theano.function([index], [cost, classifier.errors(y)], updates=grad_updates,
          givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
            y: train_set_y[index*batch_size:(index+1)*batch_size]})
    train_errors = theano.function([index], classifier.errors(y),
      givens={
        x: train_set_x[index*batch_size:(index+1)*batch_size],
        y: train_set_y[index*batch_size:(index+1)*batch_size]})
    test_pred_layers = []
    test_size = test_set_x.shape[0]
    test_layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((test_size,1,img_h,300))
    test_layer0_input = layer1.predict(test_layer0_input)
    for conv_layer in conv_layers:
        test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
        test_pred_layers.append(test_layer0_output.flatten(2))
    test_layer1_input = T.concatenate(test_pred_layers, 1)
    test_y_pred = classifier.predict(test_layer1_input)
    test_error = T.mean(T.neq(test_y_pred, y))
    test_model_all = theano.function([x,y], test_error, allow_input_downcast = True)
    test_one_sentence = theano.function([x], test_y_pred, allow_input_downcast = True)
    
    #start training over mini-batches
    print '... training'
    epoch = 0
    val_perf = 0
    test_perf = 0       
    cost_epoch = 0    
    while (epoch < n_epochs):
        epoch = epoch + 1
        train_losses = []
        print str(epoch) + "\n"
        if shuffle_batch:
            counter = 0
            for minibatch_index in np.random.permutation(range(n_train_batches)):
                counter += 1
                cost_epoch, error_epoch = train_model(minibatch_index)
                if counter % 50 == 0:
                    print "epoch %i, counter %f,  cost : %g " % (int(epoch), counter, cost_epoch)
                    dict_params = {}
                    c = 0
                    params = classifier.params
                    for conv_layer in conv_layers:
                        params += conv_layer.params
                    params += layer1.params
                    for param in params:
                        dict_params[c] = param.get_value()
                        c += 1
                    filename = str(model_name)
                    f = open(filename, "wb")
                    cPickle.dump(dict_params, f)
                    f.close()
                train_losses.append(error_epoch)
        else:
            for minibatch_index in xrange(n_train_batches):
                cost_epoch = train_model(minibatch_index)
        train_perf = 1 - np.mean(train_losses)
        # test_loss = test_model_all(test_set_x,test_set_y)
        # test_perf = 1 - test_loss
        val_losses = [val_errors(i) for i in xrange(n_val_batches)]
        val_perf = 1- np.mean(val_losses)
        print('epoch %i, train perf %f %%, val perf %f' % (epoch, train_perf * 100., val_perf*100.))
        # print('epoch %i, test perf %f' % (epoch, test_perf*100.))

    test_loss = test_model_all(test_set_x,test_set_y)        
    test_perf = 1 - test_loss
    return test_perf

def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')
        
def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        updates[exp_sg] = updates[exp_sg].astype(theano.config.floatX)
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        updates[exp_su] = updates[exp_su].astype(theano.config.floatX)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param
        updates[param] = updates[param].astype(theano.config.floatX)
    return updates 

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)
    
def safe_update(dict_to, dict_from):
    """
    re-make update dictionary for safe updating
    """
    for key, val in dict(dict_from).iteritems():
        if key in dict_to:
            raise KeyError(key)
        dict_to[key] = val
    return dict_to

def get_idx_from_sent(sent, mapping, lang, max_l=51, filter_h=5):

    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """

    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in mapping[lang]:
            x.append(mapping[lang][word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x

def make_idx_data_cv(revs, mapping, cv, max_l=51, filter_h=5):

    train, test = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], mapping, rev["language"], max_l, filter_h)
        sent.append(rev["y"])

        if rev["split"] == cv:
            test.append(sent)
        else:
            train.append(sent)

    train = np.array(train)
    test = np.array(train)
    print train.shape, test.shape
    return [train, test]

def make_idx_data_tdt(revs, mapping, max_l=51, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, dev, test = [], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], mapping, rev["language"], max_l, filter_h)
        sent.append(rev["y"])
        if len(sent) != 60:
            continue
        if rev["split"]==1:
            dev.append(sent)
        if rev["split"]==2:
            test.append(sent)
        else:
            train.append(sent)
    train = np.array(train)
    dev = np.array(dev)
    test = np.array(test)
    print train.shape, dev.shape, test.shape
    return [train, dev, test]

def parse_filter_hs(filter_hs):
    return map(int, filter_hs.split(','))
   
if __name__=="__main__":

    # Arguments for the program:
    parser = argparse.ArgumentParser(description='Convnet')
    parser.add_argument('mode')
    parser.add_argument('--filter_hs', help='filter window size', default='3,4,5')
    parser.add_argument('--epochs', help='num epochs', type=int, default=25)
    parser.add_argument('--input', default='data.p')
    parser.add_argument('--classes', default=5)
    parser.add_argument('--w2v_size', default=300)
    parser.add_argument('--params', default=None)
    parser.add_argument('--model_name', default="model")
    args = parser.parse_args()
    w2v_size = int(args.w2v_size)
    print "loading data...",
    x = cPickle.load(open(args.input,"rb"))
    revs, W, mapping = x[0], x[1], x[2]
    print "data loaded!"

    window_sizes= parse_filter_hs(args.filter_hs)
    print "window sizes", window_sizes

    num_classes = int(args.classes)
    results = []

    if args.params != None:
        file = open(args.params)
        params_loaded=cPickle.load(file)
    else:
        params_loaded = None

    datasets = make_idx_data_tdt(revs, mapping)

    perf = train_conv_net(datasets,
                          W,
                          revs,
                          str(args.model_name),
                          params_loaded = params_loaded,
                          lr_decay=0.95,
                          filter_hs=window_sizes,
                          conv_non_linear="relu",
                          hidden_units=[100,2],
                          use_valid_set=True,
                          shuffle_batch=True,
                          n_epochs=args.epochs,
                          sqr_norm_lim=9,
                          batch_size=50,
                          dropout_rate=[0.5])
    print str(perf)
