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

class LogisticRegression(object):
    def __init__(self, x, y, n_in, n_out, reg="l2"):

        #input/output matrices
        self.x = x
        self.y = y

        #creating weights matrix w
        self.w = theano.shared(
            np.zeros(
                shape=(n_in, n_out),
                dtype=self.x.dtype),
            name='w')

        #creating bias
        self.b = theano.shared(
            np.zeros(
                shape=(n_out,),
                dtype=self.w.dtype),
            name='b')

        #prob of y given x symbolic expression
        self.p_y_given_x = tensor.nnet.softmax(
            tensor.dot(self.x, self.w) + self.b)

        #prediction symbolic expression
        self.pred = tensor.argmax(self.p_y_given_x, axis=1)

        #score/erro symbolic expression (accuracy)
        self.score = tensor.mean(tensor.eq(self.pred, self.y))
        self.error = 1 - tensor.mean(tensor.eq(self.pred, self.y))

        #m = self.x.shape[0]
        #cost symbolic expression
        log_probs = tensor.log(self.p_y_given_x)
        y_indexes = tensor.arange(self.y.shape[0])
        self.neg_log_likelihood = -log_probs[y_indexes, y]
        self.cost = self.neg_log_likelihood.sum()#/m

        #regularization term symbolic expression
        if isinstance(reg, str):
            try:
                reg = _REGULARIZATIONS[reg]
            except KeyError:
                raise ValueError("'reg' must be one in [%s]" %\
                    ", ".join(_REGULARIZATIONS.keys()))
        self.reg = reg(self.w)#/m

        #model parameters
        self.params = [self.w, self.b]

        #making gradient
        self.grad = [
            (p, _grad(self.cost, p), _grad(self.reg, p))
            for p in self.params
        ]

