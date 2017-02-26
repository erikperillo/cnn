#!/usr/bin/env python3

import cnn
import mlp
import pandas as pd
import numpy as np
import theano
import sgd
from theano import tensor
import gzip
import pickle

DATA_FILEPATH = "../data/mnist.pkl.gz"

def load_data(filepath):
    with gzip.open(filepath, "rb") as f:
        try:
            dataset = pickle.load(f, encoding="latin1")
        except:
            raise

    return dataset

def main():
    print("loading data...", end="", flush=True)
    data = load_data(DATA_FILEPATH)
    print(" done")

    train_set, cv_set, test_set = data

    x_tr, y_tr = train_set
    x_cv, y_cv = cv_set
    x_te, y_te = test_set

    print("\ttrain:", x_tr.shape, y_tr.shape)
    print("\tcv:", x_cv.shape, y_cv.shape)
    print("\ttest:", x_te.shape, y_te.shape)

    x = tensor.matrix(name="x")
    y = tensor.ivector(name="y")

    layer_0_params = (
        {#conv
            "n_in_maps": 1,
            "n_out_maps": 4,
            "filter_shape": (5, 5),
        },
        {#pool
            "shape": (2, 2)
        }
    )

    layer_1_params = (
        {#conv
            "n_in_maps": 4,
            "n_out_maps": 8,
            "filter_shape": (5, 5),
        },
        {#pool
            "shape": (2, 2)
        }
    )

    fully_connected_layer_params = {
        "n_in": 8*4*4,
        "n_hidden": 32,
        "n_out": 10
    }

    batch_size = 64
    print("HMM:", x.type)
    inp = x.reshape((batch_size, 1, 28, 28))

    print("hmmm:", inp.type)
    clf = cnn.ConvolutionalNeuralNetwork(
        inp=inp,
        conv_pool_layers_params=[
            layer_0_params,
            layer_1_params],
        fully_connected_layer_params=fully_connected_layer_params)

    print("ok")

    acc = theano.function([x, y], clf.score(y))
    with_validation = True
    print("ok:", type(x_tr), type(y_tr))
    x_tr = theano.shared(np.asarray(x_tr, dtype=tensor.config.floatX),
        borrow=True)
    y_tr = theano.shared(np.asarray(y_tr, dtype=tensor.config.floatX),
        borrow=True)
    x_cv = theano.shared(np.asarray(x_cv, dtype=tensor.config.floatX),
        borrow=True)
    y_cv = theano.shared(np.asarray(y_cv, dtype=tensor.config.floatX),
        borrow=True)
    print("ok2:", type(x_tr), type(y_tr))
    def make(x, y):
        return x, tensor.cast(y, "int32")
    x_tr, y_tr = make(x_tr, y_tr)
    x_cv, y_cv = make(x_cv, y_cv)
    print("ok3:", type(x_tr), type(y_tr))
    #exit()
    #x_tr = np.array(x_tr, dtype="float64")
    #y_tr = np.array(y_tr, dtype="int32")
    if with_validation:
        print("calling sgd_with_validation", flush=True)
        #x_cv = np.array(x_cv, dtype="float64")
        #x_te = np.array(x_te, dtype="float64")
        #y_cv = np.array(y_cv, dtype="int32")
        #y_te = np.array(y_te, dtype="int32")
        sgd.sgd_with_validation(clf,
            x_tr, y_tr, x_cv, y_cv,
            learning_rate=0.01, reg_term=0.00005,
            batch_size=64, n_epochs=1000,
            max_its=5000, improv_thresh=0.01, max_its_incr=4,
            x=x,
            rel_val_tol=5e-3,
            val_freq="auto",
            verbose=True)
        print("accuracy: %.2f%%" % (100*acc(x_te, y_te)))
    else:
        print("calling sgd")
        sgd.sgd(clf, x_tr, y_tr,
            learning_rate=0.1,
            reg_term=1,
            batch_size=32,
            rel_tol=2e-3,
            n_epochs=128,
            verbose=True)
        print("accuracy: %.2f%%" % (100*acc(x_tr, y_tr)))

if __name__ == "__main__":
    main()
