import numpy as np
import theano
import sgd
from theano import tensor

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

class LogisticRegression:
    """
    Logistic Regression.
    """
    def __init__(
            self,
            inp,
            n_inp, n_out,
            reg="l2"):
        """
        Initialization.
        Parameters:
        *inp: Input tensor.
        *n_inp: Number of inputs.
        *n_out: Number of oututs.
        *reg: Regularization method for weights.
        """

        #input
        self.inp = inp

        #creating weights matrix w
        self.w = theano.shared(
            np.zeros(
                shape=(n_inp, n_out),
                dtype=self.inp.dtype),
            name='w')

        #creating bias
        self.b = theano.shared(
            np.zeros(
                shape=(n_out,),
                dtype=self.w.dtype),
            name='b')

        #model parameters
        self.params = [self.w, self.b]

        #prob of y given x symbolic expression
        self.p_y_given_x = tensor.nnet.softmax(
            tensor.dot(self.inp, self.w) + self.b)

        #prediction symbolic expression
        self.pred = tensor.argmax(self.p_y_given_x, axis=1)

        #regularization term symbolic expression
        if isinstance(reg, str):
            try:
                reg = _REGULARIZATIONS[reg]
            except KeyError:
                raise ValueError("'reg' must be one in [%s]" %\
                    ", ".join(_REGULARIZATIONS.keys()))
        self.reg = reg(self.w)

    def cost(self, y):
        """
        Symbolic expression for computing cost via negative log likelihood.
        Parameters:
        *y: output tensor.
        """
        y_indexes = tensor.arange(y.shape[0])
        log_probs = tensor.log(self.p_y_given_x)
        neg_log_likelihood = -log_probs[y_indexes, y]
        return neg_log_likelihood.sum()

    def score(self, y):
        """
        Symbolic expression for computing score (accuracy).
        Parameters:
        *y: output tensor.
        """
        return tensor.mean(tensor.eq(self.pred, y))
