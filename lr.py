import numpy as np
import theano
from theano import tensor

class LogisticRegression(object):
    def __init__(self, inp, n_in, n_out):
        #matrix for input
        self.input = inp

        #creating weights matrix w
        self.w = theano.shared(
            np.zeros(
                shape=(n_in, n_out),
                dtype=self.input.dtype),
            name='w')

        #creating bias
        self.b = theano.shared(
            np.zeros(
                shape=(n_out,),
                dtype=self.w.dtype),
            name='b')

        #output symbolic expression
        self.p_y_given_x = tensor.nnet.softmax(
            tensor.dot(self.input, self.w) + self.b)

        #prediction symbolic expression
        self.pred = tensor.argmax(self.p_y_given_x, axis=1)

    def neg_log_likelihood(self, y):
        y_indexes = tensor.arange(y.shape[0])
        log_probs = tensor.log(self.p_y_given_x)
        return -tensor.mean(log_probs[y_indexes, y])

    def score(self, y):
        """Accuracy."""
        #checking validity of arguments
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type))
        if not y.dtype.startswith('int'):
            raise ValueError("labels (y) type should be integer")

        return tensor.mean(tensor.neq(self.pred, y))
