import os
import sys
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

class HiddenLayer(object):
    def __init__(
            self, 
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
        self.input = tensor.matrix(name="input")

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
