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

class ConvolutionLayer:
    def __init__(
            self,
            inp,
            n_in_maps, n_out_maps, filter_shape,
            activation_f=tensor.nnet.sigmoid,
            w_init_f="normal",
            reg="l2",
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

        #regularization term symbolic expression
        if isinstance(reg, str):
            try:
                reg = _REGULARIZATIONS[reg]
            except KeyError:
                raise ValueError("'reg' must be one in [%s]" %\
                    ", ".join(_REGULARIZATIONS.keys()))
        self.reg = reg(self.w)

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

        #parameters
        self.params = [self.w, self.b]

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

        self.params = []

class ConvolutionalNeuralNetwork:
    def __init__(
            self,
            inp,
            conv_pool_layers_params,
            fully_connected_layer_params):

        self.inp = inp
        self.params = []

        self.conv_pool_layers = []
        for i, (conv_params, pool_params) in enumerate(conv_pool_layers_params):
            layer = {"pool": None}

            #input for layer
            layer["input"] = self.inp if i == 0 else \
                self.conv_pool_layers[i-1]["output"]

            #creating convolution layer
            conv = ConvolutionLayer(
                inp=layer["input"],
                **conv_params)
            layer["conv"] = conv

            #layer parameters
            layer["params"] = conv.params

            #creating pooling layer if required
            if pool_params is not None:
                pool = PoolingLayer(
                    inp=conv.output,
                    **pool_params)
                layer["pool"] = pool
                #updating parameters
                layer["params"] += pool.params

            #layer output
            layer["output"] = pool.output if layer["pool"] else conv.output

            self.conv_pool_layers.append(layer)
            self.params.extend(layer["params"])

        self.fully_connected_layer = mlp.MultiLayerPerceptron(
            inp=conv_pool_layers[-1]["output"].flatten(2),
            **fully_connected_layer_params)

        self.params.extend(self.fully_connected_layer.params)

        self.output = self.fully_connected_layer.output

        self.cost = self.fully_connected_layer.cost

        self.score = self.fully_connected_layer.score

        self.reg = self.fully_connected_layer.reg +\
            sum(layer["conv"].reg for layer in self.conv_pool_layers)
