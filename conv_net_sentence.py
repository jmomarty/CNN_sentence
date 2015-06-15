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
import theano.tensor as t
import warnings
import argparse
from conv_net_classes import *


warnings.filterwarnings("ignore")   


class conv_net():

    def __init__(self, wv, weights=None, s_h=60, s_w=300, reshape=30, rsh_non_linear=ReLU,
                 filter_hs=None, hidden_units=None, dropout_rate=None, batch_size=50,
                 lr_decay=0.95, conv_non_linear="relu", activations=None, sqr_norm_lim=9, rg=0.0, normalization=False):

        """
        Train a simple conv net
        img_h = sentence length (padded where necessary)
        img_w = word vector length (300 for word2vec)
        filter_hs = filter window sizes
        hidden_units = [x, y] x is the number of feature maps (per filter window), and y is the penultimate layer
        sqr_norm_lim = s^2 in the paper
        lr_decay = adadelta decay parameter
        :rtype : Float, Accuracy on the test set
        """

        self.b_s = batch_size
        self.s_h = s_h
        self.s_w = s_w

        if not filter_hs:
            filter_hs = [3, 4, 5]
        if not hidden_units:
            hidden_units = [100, 3]
        if not dropout_rate:
            dropout_rate = [0.5]
        if not activations:
            activations = ["ReLU"]
        if not weights:
            weights = range(10)
            for i in range(10):
                weights[i] = None

        rng = np.random.RandomState(3435)
        filter_w = self.s_w
        feature_maps = hidden_units[0]
        filter_shapes = []
        pool_sizes = []
        for filter_h in filter_hs:
            filter_shapes.append((feature_maps, 1, filter_h, reshape))
            pool_sizes.append((int(self.s_h-filter_h+1), int(s_w-filter_w+1)))

        print "=== Hyper-Parameters of the model ==="
        print "Batch Size: {0}".format(self.b_s)
        print "Shape of Input Sentence Matrices: ({0},{1})".format(self.s_h, s_w)
        print "Vector Reshape: {0} => {1}".format(s_w, reshape)
        print "Non-Linearity on Reshape Layer: {0}".format(rsh_non_linear)
        print "Number of Filters: {0}".format(len(filter_hs))
        print "Shape of Filters: {0}".format(filter_hs)
        print "Non-Linearity on Convolution Layers: {0}".format(conv_non_linear)
        print "Number of Feature Maps per Filter: {0}".format(hidden_units[0])
        print "Output Vector Size: {0}".format(hidden_units[1])
        print "Dropout: {0}, Learning Rate Decay: {1}, Square Norm Lim: {2}".format(dropout_rate, lr_decay, sqr_norm_lim)
        print "Batch Normalization: {0}".format(normalization)

        # define model architecture
        self.sent = t.matrix('x')
        self.label = t.ivector('y')
        self.words = theano.shared(value=np.asarray(wv, dtype=theano.config.floatX), name="Words")
        layer0_input = self.words[t.cast(self.sent.flatten(), dtype="int32")].reshape((self.sent.shape[0], 1, self.sent.shape[1], s_w))
        self.layer1 = HiddenLayer(rng, layer0_input, n_in=s_w, n_out=reshape, activation=ReLU,
                             w=weights[8], b=weights[9], use_bias=True)
        layer1_input = self.layer1.output
        self.conv_layers = []
        layer2_inputs = []
        for i in xrange(len(filter_hs)):
            c = 2*(len(filter_hs)-i)+1
            filter_shape = filter_shapes[i]
            pool_size = pool_sizes[i]
            conv_layer = LeNetConvPoolLayer(rng, ipt=layer1_input, image_shape=(self.b_s, 1, self.s_h, reshape),
                                            filter_shape=filter_shape, params_loaded=[weights[c-1], weights[c]],
                                            name_model="cn_"+str(i), poolsize=pool_size, non_linear=conv_non_linear)
            layer2_input = conv_layer.output.flatten(2)
            self.conv_layers.append(conv_layer)
            layer2_inputs.append(layer2_input)
        layer2_input = t.concatenate(layer2_inputs, 1)
        hidden_units[0] = feature_maps*len(filter_hs)
        self.classifier = MLPDropout(rng, ipt=layer2_input, layer_sizes=hidden_units, activations=activations,
                                dropout_rates=dropout_rate, params=[weights[0], weights[1]])

        # define parameters of the model and update functions using adadelta
        self.params = self.classifier.params
        for i in range(len(self.conv_layers)):
            self.params += self.conv_layers[len(self.conv_layers)-1-i].params
        self.params += self.layer1.params
        self.cost = self.classifier.negative_log_likelihood(self.label, rg)
        dropout_cost = self.classifier.dropout_negative_log_likelihood(self.label, rg)
        self.grad_updates = sgd_updates_adadelta(self.params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)

    def fit(self, dst, model_name, info_display_freq=50, save_freq=50, n_epochs=25, use_valid_set=True, shuffle_batch=True):

        # shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate
        # extra data (at random)

        index = t.lscalar()

        np.random.seed(3435)
        if dst[0].shape[0] % self.b_s > 0:
            extra_data_num = self.b_s - dst[0].shape[0] % self.b_s
            train_set = np.random.permutation(dst[0])
            extra_data = train_set[:extra_data_num]
            new_data = np.append(dst[0], extra_data, axis=0)
        else:
            new_data = dst[0]
        new_data = np.random.permutation(new_data)
        n_batches = new_data.shape[0]/self.b_s
        n_train_batches = int(np.round(n_batches*0.9))
        print "Number of Train Batches : " + str(n_train_batches)
        if len(dst) == 3:
            print "Using Train/Dev/Test.."
            train_set = new_data
            val_set = dst[1]
            train_set_x, train_set_y = shared_dataset((train_set[:, :self.s_h], train_set[:, -1]))
            val_set_x, val_set_y = shared_dataset((val_set[:, :self.s_h], val_set[:, -1]))
            test_set_x = dst[2][:, :self.s_h]
            test_set_y = np.asarray(dst[2][:, -1], "int32")
            n_val_batches = int(val_set.shape[0] / self.b_s)
            print "Number of Validation Batches : " + str(n_val_batches)
            val_errors = theano.function([index], self.classifier.errors(self.label),
                                         givens={self.sent: val_set_x[index * self.b_s: (index + 1) * self.b_s],
                                                 self.label: val_set_y[index * self.b_s: (index + 1) * self.b_s]})
        else:
            test_set_x = dst[1][:, :self.s_h]
            test_set_y = np.asarray(dst[1][:, -1], "int32")
            if use_valid_set:
                train_set = new_data[:n_train_batches*self.b_s, :]
                val_set = new_data[n_train_batches*self.b_s:, :]
                train_set_x, train_set_y = shared_dataset((train_set[:, :self.s_h], train_set[:, -1]))
                val_set_x, val_set_y = shared_dataset((val_set[:, :self.s_h], val_set[:, -1]))
                n_val_batches = n_batches - n_train_batches
                print "Number of Validation Batches : " + str(n_val_batches)
                val_errors = theano.function([index], self.classifier.errors(self.label),
                                             givens={self.sent: val_set_x[index * self.b_s: (index + 1) * self.b_s],
                                                     self.label: val_set_y[index * self.b_s: (index + 1) * self.b_s]})
            else:
                train_set = new_data[:, :]
                train_set_x, train_set_y = shared_dataset((train_set[:, :self.s_h], train_set[:, -1]))

        train_model = theano.function([index], [self.cost, self.classifier.errors(self.label)], updates=self.grad_updates,
                                      givens={self.sent: train_set_x[index*self.b_s:(index+1)*self.b_s],
                                              self.label: train_set_y[index*self.b_s:(index+1)*self.b_s]})
        test_pred_layers = []
        test_size = test_set_x.shape[0]
        test_layer0_input = self.words[t.cast(self.sent.flatten(), dtype="int32")].reshape((test_size, 1, self.s_h, self.s_w))
        test_layer0_input = self.layer1.predict(test_layer0_input)
        for conv_layer in self.conv_layers:
            test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
            test_pred_layers.append(test_layer0_output.flatten(2))
        test_layer1_input = t.concatenate(test_pred_layers, 1)
        test_y_pred = self.classifier.predict(test_layer1_input)
        test_error = t.mean(t.neq(test_y_pred, self.label))
        test_model_all = theano.function([self.sent, self.label], test_error, allow_input_downcast=True)

        # start training over mini-batches
        print 'Training over mini-batches...'
        epoch = 0
        while epoch < n_epochs:
            epoch += 1
            train_losses = []
            print str(epoch) + "\n"
            counter = 0
            if shuffle_batch:
                for minibatch_index in np.random.permutation(range(n_train_batches)):
                    counter += 1
                    cost_epoch, error_epoch = train_model(minibatch_index)
                    if counter % info_display_freq == 0:
                        print "Epoch {0}, Counter {1},  Cost : {2} ".format(int(epoch), int(counter), cost_epoch)
                    if counter % save_freq == 0:
                        save_params(self.classifier, self.conv_layers, self.layer1, model_name)
                    train_losses.append(error_epoch)
            else:
                for minibatch_index in xrange(n_train_batches):
                    counter += 1
                    cost_epoch, error_epoch = train_model(minibatch_index)
                    train_losses.append(error_epoch)
                    if counter % info_display_freq == 0:
                        print "Epoch {0}, Counter {1},  Cost : {2} ".format(int(epoch), int(counter), cost_epoch)
                    if counter % save_freq == 0:
                        save_params(self.classifier, self.conv_layers, self.layer1, model_name)
            train_perf = 1 - np.mean(train_losses)
            test_loss = test_model_all(test_set_x, test_set_y)
            test_perf = 1 - test_loss
            if use_valid_set:
                val_losses = [val_errors(i) for i in xrange(n_val_batches)]
                val_perf = 1 - np.mean(val_losses)
                print 'Epoch {0}: Train Perf {1}%, Val Perf {2}%, Test Perf {3}%'.format(epoch, train_perf * 100.,
                                                                                         val_perf*100., test_perf*100)
            else:
                print 'Epoch {0}: Train Perf {1}%, Test Perf {2}%'.format(epoch, train_perf * 100., test_perf*100)
        test_loss = test_model_all(test_set_x, test_set_y)
        test_perf = 1 - test_loss
        return test_perf

    def predict(self, dst):

        test_set_x = dst[1][:, :self.s_h]

        test_pred_layers = []
        test_size = test_set_x.shape[0]
        test_layer0_input = self.words[t.cast(self.sent.flatten(), dtype="int32")].reshape((test_size, 1, self.s_h, self.s_w))
        test_layer0_input = self.layer1.predict(test_layer0_input)
        for conv_layer in self.conv_layers:
            test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
            test_pred_layers.append(test_layer0_output.flatten(2))
        test_layer1_input = t.concatenate(test_pred_layers, 1)
        test_y_pred = self.classifier.predict(test_layer1_input)
        test_predict_all = theano.function([self.sent], test_y_pred, allow_input_downcast=True)
        output = open("prediction.txt", "wb")
        for x in test_predict_all(test_set_x):
            output.write(str(x)+"\r")

    def accuracy(self, dst):

        test_set_x = dst[1][:, :self.s_h]
        test_set_y = np.asarray(dst[1][:, -1], "int32")

        test_pred_layers = []
        test_size = test_set_x.shape[0]
        test_layer0_input = self.words[t.cast(self.sent.flatten(), dtype="int32")].reshape((test_size, 1, self.s_h, self.s_w))
        test_layer0_input = self.layer1.predict(test_layer0_input)
        for conv_layer in self.conv_layers:
            test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
            test_pred_layers.append(test_layer0_output.flatten(2))
        test_layer1_input = t.concatenate(test_pred_layers, 1)
        test_y_pred = self.classifier.predict(test_layer1_input)
        test_error = t.mean(t.neq(test_y_pred, self.label))
        test_model_all = theano.function([self.sent, self.label], test_error, allow_input_downcast=True)
        return 1 - test_model_all(test_set_x, test_set_y)

def save_params(classifier, conv_layers, layer1, model_name):

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
    save = open(filename, "wb")
    cPickle.dump(dict_params, save)
    save.close()


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
    return shared_x, t.cast(shared_y, 'int32')


def sgd_updates_adadelta(params, cost, rho=0.95, epsilon=1e-6, norm_lim=9):

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
        exp_sqr_grads[param] = theano.shared(value=as_float_x(empty), name="exp_grad_{0}".format(param.name))
        gp = t.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_float_x(empty), name="exp_grad_{0}".format(param.name))
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * t.sqr(gp)
        updates[exp_sg] = up_exp_sg
        updates[exp_sg] = updates[exp_sg].astype(theano.config.floatX)
        step = -(t.sqrt(exp_su + epsilon) / t.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * t.sqr(step)
        updates[exp_su] = updates[exp_su].astype(theano.config.floatX)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name != 'Words'):
            col_norms = t.sqrt(t.sum(t.sqr(stepped_param), axis=0))
            desired_norms = t.clip(col_norms, 0, t.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param
        updates[param] = updates[param].astype(theano.config.floatX)
    return updates 


def as_float_x(variable):
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


def get_idx_from_sent(sent, mpg, lang, max_l=51, filter_h=5):

    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """

    s = []
    pad = filter_h - 1
    for i in xrange(pad):
        s.append(0)
    words = sent.split()
    for word in words:
        if word in mpg[lang]:
            s.append(mpg[lang][word])
        if len(s) > max_l:
            break
    while len(s) < max_l+2*pad:
        s.append(0)
    return s


def make_idx_data_cv(rvs, mpg, cv, max_l=51, filter_h=5):

    train, test = [], []
    for rev in rvs:
        sent = get_idx_from_sent(rev["text"], mpg, rev["language"], max_l, filter_h)
        sent.append(rev["y"])
        if rev["split"] == cv:
            test.append(sent)
        else:
            train.append(sent)
    train = np.array(train)
    test = np.array(test)
    print train.shape, test.shape
    return [train, test]


def make_idx_data_tdt(rvs, mpg, max_l=51, filter_h=5):

    """
    Transforms sentences into a 2-d matrix.
    """

    train, dev, test = [], [], []
    for rev in rvs:
        sent = get_idx_from_sent(rev["text"], mpg, rev["language"], max_l, filter_h)
        sent.append(rev["y"])

        if rev["split"] == 1:
            dev.append(sent)
        if rev["split"] == 2:
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


if __name__ == "__main__":

    # Arguments for the program:
    parser = argparse.ArgumentParser(description='Convnet')
    parser.add_argument('mode', help='cv/dev/inf')
    parser.add_argument('--folds', default=10)
    parser.add_argument('--filter_hs', help='filter window size', default='3,4,5')
    parser.add_argument('--epochs', help='num epochs', type=int, default=25)
    parser.add_argument('--input', default='data.p')
    parser.add_argument('--classes', default=5)
    parser.add_argument('--w2v_size', default=300)
    parser.add_argument('--params', default=None)
    parser.add_argument('--model_name', default="model")
    parser.add_argument('--words', default=None)
    parser.add_argument('--dropout', default=0.5)
    parser.add_argument('--reg', default=0.0)
    parser.add_argument('--max_l', default=51)
    parser.add_argument('--batch_size', default=50)
    parser.add_argument('--info_display', default=50)
    parser.add_argument('--inference', default=False)
    args = parser.parse_args()
    w2v_size = int(args.w2v_size)

    print "Loading Data...\r"
    x = cPickle.load(open(args.input, "rb"))
    if args.words is None:
        revs, W, mapping = x[0], x[1], x[2]
    else:
        revs, _, _ = x[0], x[1], x[2]
        y = cPickle.load(open(args.words, "rb"))
        _, W, mapping = y[0], y[1], y[2]
    print "Data Loaded!\n"

    window_sizes = parse_filter_hs(args.filter_hs)

    num_classes = int(args.classes)
    results = []

    if args.params is not None:
        f = open(args.params)
        params_loaded = cPickle.load(f)
    else:
        params_loaded = None

    max_l = int(args.max_l)
    pad_l = int(max(args.filter_hs))
    max_sent_length = max_l + 2*(pad_l - 1) - 1
    b_s = int(args.batch_size)
    i_d = int(args.info_display)

    if args.mode == 'dev':  # Train/Dev/Test Split
        datasets = make_idx_data_tdt(revs, mapping, max_l=int(args.max_l))
        cnn = conv_net(W,
                       weights=params_loaded,
                       filter_hs=window_sizes,
                       hidden_units=[100, num_classes],
                       dropout_rate=[float(args.dropout)],
                       rg=float(args.reg),
                       s_h=max_sent_length,
                       batch_size=b_s,
                       s_w=w2v_size
                       )
        perf = cnn.fit(datasets,
                       str(args.model_name),
                       info_display_freq=i_d,
                       n_epochs=args.epochs
                       )
        print "Test Set Accuracy = {0}% \n".format(str(perf))
        print "End of Training."
    elif args.mode == 'cv':  # Cross-Validation
        for i in xrange(int(args.folds)):
            print "Cross-Validation: Step {0} out of {1}\n".format(i, int(args.folds))
            datasets = make_idx_data_cv(revs, mapping, i, max_l=int(args.max_l))
            cnn = conv_net(W,
                           weights=params_loaded,
                           filter_hs=window_sizes,
                           hidden_units=[100, num_classes],
                           dropout_rate=[float(args.dropout)],
                           rg=float(args.reg),
                           s_h=max_sent_length,
                           batch_size=b_s,
                           s_w=w2v_size
                           )
            perf = cnn.fit(datasets,
                           str(args.model_name),
                           info_display_freq=i_d,
                           n_epochs=args.epochs
                           )
            print "Cross-Validation: Step {0} out of {1}... Test Accuracy = {2}% \n".format(i, args.mode, str(perf))
            results.append(perf)
        print "Cross-Validation: Average Accuracy = {0}% \n".format(str(np.mean(results)))
        print "End of Training."
    elif args.mode == 'inf':
        print "Inference\n"
        datasets = make_idx_data_cv(revs, mapping, 0, max_l=int(args.max_l))
        cnn = conv_net(W,
                       weights=params_loaded,
                       filter_hs=window_sizes,
                       hidden_units=[100, num_classes],
                       dropout_rate=[float(args.dropout)],
                       rg=float(args.reg),
                       s_h=max_sent_length,
                       batch_size=b_s,
                       s_w=w2v_size
                       )
        cnn.predict(datasets)
    elif args.mode == 'acc':
        print "Compute Accuracy\n"
        datasets = make_idx_data_cv(revs, mapping, 0, max_l=int(args.max_l))
        cnn = conv_net(W,
                       weights=params_loaded,
                       filter_hs=window_sizes,
                       hidden_units=[100, num_classes],
                       dropout_rate=[float(args.dropout)],
                       rg=float(args.reg),
                       s_h=max_sent_length,
                       batch_size=b_s,
                       s_w=w2v_size
                       )
        print "Accuracy = {0}".format(cnn.accuracy(datasets))
