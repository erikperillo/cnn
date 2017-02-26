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

def cvt_type(xy_set, x_dtype="float64", y_dtype="int32"):
    x_set, y_set = xy_set
    return np.array(x_set, dtype=x_dtype), np.array(y_set, dtype=y_dtype)

def load_data(filepath):
    with gzip.open(filepath, "rb") as f:
        try:
            dataset = pickle.load(f, encoding="latin1")
        except:
            raise

    return tuple(map(cvt_type, dataset))

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
            "n_inp_maps": 1,
            "inp_maps_shape": (28, 28),
            "n_out_maps": 5,
            "filter_shape": (5, 5),
        },
        {#pool
            "shape": (2, 2)
        }
    )

    layer_1_params = (
        {#conv
            #"n_in_maps": 4,
            "n_out_maps": 10,
            "filter_shape": (5, 5),
        },
        {#pool
            "shape": (2, 2)
        }
    )

    fully_connected_layer_params = {
        #"n_inp": 10*4*4,
        "n_hidden": 64,
        "n_out": 10
    }

    batch_size = 32
    bs = tensor.iscalar()
    #inp = x.reshape((batch_size, 1, 28, 28))
    #inp = x.reshape((bs, 1, 28, 28))
    inp = x.reshape((x.shape[0], 1, 28, 28))
    #print(inp.shape[1].value)

    clf = cnn.ConvolutionalNeuralNetwork(
        #inp=inp,
        inp=x,
        conv_pool_layers_params=[
            layer_0_params,
            layer_1_params],
        fully_connected_layer_params=fully_connected_layer_params)
    #exit()

    acc = theano.function([x, y], clf.score(y))
    #exit()

    with_validation = True

    x_tr_sh = theano.shared(x_tr, borrow=True)
    y_tr_sh = theano.shared(y_tr, borrow=True)
    x_cv_sh = theano.shared(x_cv, borrow=True)
    y_cv_sh = theano.shared(y_cv, borrow=True)

    #print(np.mean(clf.params[0]), np.std(clf.params[0]))
    print(clf.params[0].get_value(borrow=True).mean())
    if with_validation:
        print("calling sgd_with_validation", flush=True)
        sgd.sgd_with_validation(clf,
            x_tr_sh, y_tr_sh, x_cv_sh, y_cv_sh,
            learning_rate=0.0001, reg_term=0.00001,
            batch_size=250, n_epochs=32,
            max_its=20000, improv_thresh=0.01, max_its_incr=4,
            x=x,
            rel_val_tol=5e-3,
            val_freq="auto",
            verbose=True)
    else:
        print("calling sgd")
        sgd.sgd(clf, x_tr, y_tr,
            learning_rate=0.1,
            reg_term=1,
            batch_size=32,
            rel_tol=2e-3,
            n_epochs=128,
            verbose=True)

    print(clf.params[0].get_value(borrow=True).mean())
    print("accuracy: %.2f%%" % (100*acc(x_tr[:batch_size*4,:],
        y_tr[:batch_size*4])))

if __name__ == "__main__":
    main()
