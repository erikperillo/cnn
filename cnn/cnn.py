import numpy as np
import theano
from theano import tensor
from theano.tensor.signal import pool
import mlp
import lr

def _dim_after_conv(size, k_size, stride, ignore_border=True):
    if k_size > size:
        return 0
    _size = size - k_size
    return 1 + _size//stride + int(not ignore_border)*int(_size%stride != 0)

def _dim_after_conv_nd(sizes, k_sizes, strides, ignore_border=True):
    return tuple(_dim_after_conv(k, s, st, ignore_border)\
        for k, s, st in zip(sizes, k_sizes, strides))

def _get_rng(rand_state=None):
    if isinstance(rand_state, int) or rand_state is None:
        return np.random.RandomState(rand_state)
    return rand_state

def _uniform_w_init(w_shape, n_inp_maps, n_out_maps, filter_shape,
        rand_state=42):
    rng = _get_rng(rand_state)
    filter_h, filter_w = filter_shape
    bound = np.sqrt(n_inp_maps*filter_w*filter_h)
    return rng.uniform(low=-1/bound, high=1/bound, size=w_shape)

def _normal_w_init(w_shape, n_inp_maps, n_out_maps, filter_shape,
        rand_state=42):
    rng = _get_rng(rand_state)
    filter_h, filter_w = filter_shape
    std = np.sqrt(n_inp_maps*filter_w*filter_h)
    return rng.normal(loc=0, scale=std, size=w_shape)

#Weight initialization methods for convolution layer.
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

#Regularization methods for weights.
_REGULARIZATIONS = {
    "none": _zero,
    "l1": _l1,
    "l2": _l2
}

class ConvolutionLayer:
    """
    Convolution layer for Convolutional Neural Network.
    """
    def __init__(
            self,
            inp,
            n_inp_maps, inp_maps_shape,
            n_out_maps, filter_shape,
            activation_f=tensor.nnet.sigmoid,
            w_init_f="normal",
            reg="l2",
            rand_state=42):
        """
        Initialization of layer.
        Parameters:
        *inp: Input tensor.
        *n_inp_maps: Number of input maps.
        *inp_maps_shape: Input maps shape in format (rows, cols).
        *n_out_maps: Number of output maps.
        *activation_f: Activation function to use.
        *w_init_f: Weight initialization method.
        *reg: Regularization method for weights.
        *rand_state: Random state.
        """

        #checking validity of input method
        if isinstance(w_init_f, str):
            try:
                w_init_f = _W_INIT_METHODS[w_init_f]
            except KeyError:
                raise ValueError("'w_init_method' must be one in [%s]" %\
                    ", ".join(_W_INIT_METHODS.keys()))

        #input
        self.inp = inp

        #storing parameters
        self.n_inp_maps = n_inp_maps
        self.n_out_maps = n_out_maps
        self.inp_maps_shape = inp_maps_shape
        self.filter_shape = filter_shape

        #creating weights tensor w
        filter_h, filter_w = filter_shape
        w_shape = (n_out_maps, n_inp_maps, filter_h, filter_w)
        self.w = theano.shared(
            np.asarray(
                w_init_f(
                    w_shape,
                    n_inp_maps, n_out_maps, filter_shape,
                    rand_state),
                dtype=self.inp.dtype),
            borrow=True,
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
        self.b = theano.shared(
            np.asarray(
                _get_rng(rand_state).uniform(
                    low=-0.5,
                    high=0.5,
                    size=b_shape),
                dtype=self.inp.dtype),
            borrow=True,
            name="b")

        #parameters
        self.params = [self.w, self.b]

        #symbolic expression for convolution
        conv_output = tensor.nnet.conv2d(self.inp, self.w)

        #symbolic expression for final output
        self.output = activation_f(
            conv_output + self.b.dimshuffle("x", 0, "x", "x"))

class PoolingLayer:
    """
    Pooling layer for Convolutional Neural Network.
    """
    def __init__(
            self,
            inp,
            shape,
            stride=None,
            mode="max",
            ignore_border=True):
        """
        Initialization of layer.
        Parameters:
        *inp: Input tensor.
        *shape: Pooling shape in form (rows, cols).
        *stride: Pooling stride. If None, stride = shape.
        *mode: Pooling mode.
        *ignore_border: Ignore border.
        """

        #input
        self.inp = inp

        #attributes
        self.shape = shape
        self.ignore_border = ignore_border
        self.stride = shape if stride is None else stride

        #parameters
        self.params = []

        #symbolic expression for pooling
        self.output = pool.pool_2d(self.inp,
            ds=shape,
            ignore_border=ignore_border,
            st=stride)

class ConvolutionalNeuralNetwork:
    """
    Convolutinal Neural Network.
    """
    def __init__(
            self,
            inp,
            conv_pool_layers_params,
            fully_connected_layer_params):
        """
        Initialization of network.
        Parameters:
        *inp: Either a 2D tensor (to be transformed into 4D later)
            or a 4D tensor in format (n_batches, n_inp_maps, maps_h, maps_w).
        *conv_pool_layers_params: List of tuples in format (conv, pool).
            Each tuple represents a convolution-pooling layer.
            conv is a dict with arguments for ConvolutionLayer class.
            pool is a dict with arguments for PoolingLayer class (can be None).
            Some parameters can be inferred, e.g. number of input maps for conv
            layer n+1 (since number of output maps for conv layer n is known).
        *fully_connected_layer_params: Dictionary for parameters for class
            mlp.MultiLayerPerceptron. n_inp parameter can be inferred.
        """

        inp_maps_shape = conv_pool_layers_params[0][0]["inp_maps_shape"]

        #setting up input
        if inp.ndim != 4:
            print("reshaping...")
            n_inp_maps = conv_pool_layers_params[0][0]["n_inp_maps"]
            inp = inp.reshape((inp.shape[0], n_inp_maps) + inp_maps_shape)
        self.inp = inp

        #initial parameters
        self.params = []

        #conv-pool layers
        self.conv_pool_layers = []

        #this tuple keeps track of outputs shape in last conv-pool layer
        end_maps_hw = inp_maps_shape

        #adding layers
        for i, (conv_params, pool_params) in enumerate(conv_pool_layers_params):
            #current layer dict
            layer = {"pool": None}

            #input for layer
            layer["input"] = self.inp if i == 0 else \
                self.conv_pool_layers[i-1]["output"]

            #inferring some conv parameters
            if not "n_inp_maps" in conv_params and i > 0:
                conv_params["n_inp_maps"] = \
                    self.conv_pool_layers[-1]["conv"].n_out_maps
            if not "inp_maps_shape" in conv_params and i > 0:
                conv_params["inp_maps_shape"] = end_maps_hw

            #creating convolution layer
            conv = ConvolutionLayer(
                inp=layer["input"],
                **conv_params)
            layer["conv"] = conv

            #updating final output shape
            end_maps_hw = _dim_after_conv_nd(
                end_maps_hw, conv.filter_shape, (1, 1))

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

                #updating final output shape
                end_maps_hw = _dim_after_conv_nd(
                    end_maps_hw, pool.shape, pool.stride)

            #layer output
            layer["output"] = pool.output if layer["pool"] else conv.output

            #appending layer dict to list of conv-pool layers
            self.conv_pool_layers.append(layer)

            #updating parameters list for model
            self.params.extend(layer["params"])

        #inferring n_inp if necessary
        if not "n_inp" in fully_connected_layer_params:
            n_maps_inp = self.conv_pool_layers[-1]["conv"].n_out_maps
            n_inps_per_map = end_maps_hw[0]*end_maps_hw[1]
            fully_connected_layer_params["n_inp"] = n_inps_per_map*n_maps_inp

        #creating fully connected layer
        self.fully_connected_layer = mlp.MultiLayerPerceptron(
            inp=self.conv_pool_layers[-1]["output"].flatten(2),
            **fully_connected_layer_params)

        #updating parameters
        self.params.extend(self.fully_connected_layer.params)

        #model cost(y) function
        self.cost = self.fully_connected_layer.cost

        #model score(y) function
        self.score = self.fully_connected_layer.score

        #pred function
        self.pred = self.fully_connected_layer.pred

        #model regularization function
        self.reg = self.fully_connected_layer.reg +\
            sum(layer["conv"].reg for layer in self.conv_pool_layers)
