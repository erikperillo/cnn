import os
import sys
import lr
import timeit
import numpy as np
import theano
from theano import tensor

def _get_rng(rand_state=None):
    if isinstance(rand_state, int) or rand_state is None:
        return np.random.RandomState(rand_state)
    return rand_state

def _xavier_tanh_w_init(n_in, n_out, rand_state=42):
    rng = _get_rng(rand_state)
    bound = np.sqrt(6/(n_in + n_out))
    return rng.uniform(low=-bound, high=bound, size=(n_in, n_out))

def _xavier_sigmoid_w_init(n_in, n_out, rand_state=42):
    rng = _get_rng(rand_state)
    bound = 4*np.sqrt(6/(n_in + n_out))
    return rng.uniform(low=-bound, high=bound, size=(n_in, n_out))

_W_INIT_METHODS = {
    "xavier_tanh": _xavier_sigmoid_w_init,
    "xavier_sigmoid": _xavier_sigmoid_w_init
}

def _zero(x):
    return (x*0).sum()

def _l1(x):
    return abs(x).sum()

def _l2(x):
    return (x**2).sum()

_REGULARIZATIONS = {
    "none": _zero,
    "l1": _l1,
    "l2": _l2
}

def _grad(cost, wrt):
    try:
        return tensor.grad(cost=cost, wrt=wrt)
    except theano.gradient.DisconnectedInputError:
        return 0

class HiddenLayer(object):
    def __init__(
            self, 
            inp,
            n_in, n_out,
            activation_f=tensor.tanh,
            w_init_f="xavier_tanh",
            rand_state=42):

        #checking validity of input method
        if isinstance(w_init_f, str):
            try:
                w_init_f = _W_INIT_METHODS[w_init_f]
            except KeyError:
                raise ValueError("'w_init_method' must be one in [%s]" %\
                    ", ".join(_W_INIT_METHODS.keys()))

        #matrix for input
        self.input = inp

        #creating weights matrix w
        self.w = theano.shared(
            np.asarray(
                w_init_f(n_in, n_out),
                dtype=self.input.dtype),
            name="w")

        #creating bias
        self.b = theano.shared(
            np.asarray(
                _get_rng(rand_state).uniform(
                    low=-0.5,
                    high=0.5,
                    size=(n_out,)),
                dtype=self.input.dtype),
            name="b")

        #symbolic expression for output
        linear_output = tensor.dot(self.input, self.w) + self.b
        self.output = activation_f(linear_output)

        #theano function to compute output
        self.f = theano.function([self.input], self.output)

        self.params = [self.w, self.b]

class MultiLayerPerceptron:
    def __init__(
            self, 
            x, y,
            n_in, n_hidden, n_out,
            activation_f=tensor.tanh,
            w_init_f="xavier_tanh",
            reg="l2",
            rand_state=42):

        #input/output matrices
        self.x = x
        self.y = y

        #hidden layer
        self.hidden_layer = HiddenLayer(
            inp=x,
            n_in=n_in, n_out=n_hidden,
            activation_f=activation_f)

        #logistic regression layer
        self.log_reg_layer = lr.LogisticRegression(
            x=self.hidden_layer.output, y=y,
            n_in=n_hidden, n_out=n_out)

        #score symbolic expression (accuracy)
        self.score = self.log_reg_layer.score

        m = self.x.shape[0]
        #cost symbolic expression
        self.cost = self.log_reg_layer.cost

        #regularization term symbolic expression
        if isinstance(reg, str):
            try:
                reg = _REGULARIZATIONS[reg]
            except KeyError:
                raise ValueError("'reg' must be one in [%s]" %\
                    ", ".join(_REGULARIZATIONS.keys()))
        self.reg = (reg(self.hidden_layer.w) + reg(self.log_reg_layer.w))/m

        #model parameters
        self.params = self.hidden_layer.params + self.log_reg_layer.params

        #making gradient
        self.grad = [
            (p, _grad(self.cost, p), _grad(self.reg, p))
            for p in self.params
        ]
