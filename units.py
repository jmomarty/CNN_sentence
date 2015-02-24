import theano
import theano.tensor as T

#different non-linearities
def ReLU(x):
    return T.maximum(0.0, x)
def Sigmoid(x):
    return T.nnet.sigmoid(x)
def Tanh(x):
    return T.tanh(x)
def Iden(x):
    return x
       
