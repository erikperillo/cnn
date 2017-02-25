import numpy as np
import theano
from theano import tensor
from theano.tensor.signal import pool

def _get_rng(rand_state=None):
    if isinstance(rand_state, int) or rand_state is None:
        return np.random.RandomState(rand_state)
    return rand_state

def _uniform_w_init(w_shape, n_in_maps, n_out_maps, filter_shape,
        rand_state=42):
    rng = _get_rng(rand_state)
    filter_h, filter_w = filter_shape
    bound = np.sqrt(n_in_maps*filter_w*filter_h)
    return rng.uniform(low=-1/bound, high=1/bound, size=w_shape)

def _normal_w_init(w_shape, n_in_maps, n_out_maps, filter_shape,
        rand_state=42):
    rng = _get_rng(rand_state)
    filter_h, filter_w = filter_shape
    std = np.sqrt(n_in_maps*filter_w*filter_h)
    return rng.normal(loc=0, scale=std, size=w_shape)

_W_INIT_METHODS = {
    "normal": _normal_w_init,
    "uniform": _uniform_w_init
}

class ConvolutionLayer:
    def __init__(
            self,
            inp,
            n_in_maps, n_out_maps, filter_shape,
            activation_f=tensor.nnet.sigmoid,
            w_init_f="normal",
            rand_state=42):

        #checking validity of input method
        if isinstance(w_init_f, str):
            try:
                w_init_f = _W_INIT_METHODS[w_init_f]
            except KeyError:
                raise ValueError("'w_init_method' must be one in [%s]" %\
                    ", ".join(_W_INIT_METHODS.keys()))

        #4D tensor for input
        self.input = inp

        #creating weights tensor w
        filter_h, filter_w = filter_shape
        w_shape = (n_out_maps, n_in_maps, filter_h, filter_w)
        self.w = theano.shared(
            np.asarray(
                w_init_f(
                    w_shape,
                    n_in_maps, n_out_maps, filter_shape,
                    rand_state),
                dtype=self.input.dtype),
            name="w")

        #creating bias
        b_shape = (n_out_maps,)
        b = theano.shared(
            np.asarray(
                _get_rng(rand_state).uniform(
                    low=-0.5,
                    high=0.5,
                    size=b_shape),
                dtype=self.input.dtype),
            name="b")
        self.b = b.dimshuffle("x", 0, "x", "x")

        #symbolic expression for convolution
        conv_output = tensor.nnet.conv2d(self.input, self.w)

        #symbolic expression for final output
        self.output = activation_f(conv_output + self.b)

        #theano function to compute actual filtering
        self.f = theano.function([self.input], self.output)

_POOLING_METHODS = {
    "max": pool.pool_2d
}

class PoolingLayer:
    def __init__(
            self,
            inp,
            shape,
            pool_f="max",
            ignore_border=True):
        #checking validity of input method
        if isinstance(pool_f, str):
            try:
                pool_f = _POOLING_METHODS[pool_f]
            except KeyError:
                raise ValueError("'method' must be one in [%s]" %\
                    ", ".join(_POOLING_METHODS.keys()))

        self.shape = shape

        #4D tensor for input
        self.input = inp

        #symbolic expression for pooling
        self.output = pool_f(self.input, shape, ignore_border=ignore_border)

        #theano function to compute pooling
        self.f = theano.function([self.input], self.output)
