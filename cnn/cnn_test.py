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

    """with open("/home/erik/db/data.pkl", "rb") as f:
        x, y = pickle.load(f)

    tr = 300
    x = np.array(x, dtype="float64")
    y = np.array(y, dtype="int32")
    x_tr, y_tr = x[:tr, :], y[:tr]
    x_cv, y_cv = x[tr:, :], y[tr:]
    x_te, y_te = x_cv, y_cv"""

    print("\ttrain:", x_tr.shape, y_tr.shape)
    print("\tcv:", x_cv.shape, y_cv.shape)
    print("\ttest:", x_te.shape, y_te.shape)

    x = tensor.matrix(name="x")
    y = tensor.ivector(name="y")

    layer_0_params = (
        {#conv
            "n_inp_maps": 1,
            "inp_maps_shape": (28, 28),
            #"inp_maps_shape": (48, 32),
            "n_out_maps": 5,
            #"n_out_maps": 4,
            "filter_shape": (7, 7),
        },
        {#pool
            "shape": (2, 2)
        }
    )

    layer_1_params = (
        {#conv
            #"n_in_maps": 4,
            "n_out_maps": 10,
            #"n_out_maps": 6,
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

    inp = x.reshape((x.shape[0], 1, 28, 28))
    #inp = x.reshape((x.shape[0], 1, 48, 32))

    load = False
    if load:
        print("loading model...")
        with open("cnn_model.pkl", "rb") as f:
            clf = pickle.load(f)
    else:
        clf = cnn.ConvolutionalNeuralNetwork(
            #inp=inp,
            inp=x,
            conv_pool_layers_params=[
                layer_0_params,
                layer_1_params],
            fully_connected_layer_params=fully_connected_layer_params)


        with_validation = True

        x_tr_sh = theano.shared(x_tr, borrow=True)
        y_tr_sh = theano.shared(y_tr, borrow=True)
        x_cv_sh = theano.shared(x_cv, borrow=True)
        y_cv_sh = theano.shared(y_cv, borrow=True)

        if with_validation:
            print("calling sgd_with_validation", flush=True)
            sgd.sgd_with_validation(clf,
                x_tr_sh, y_tr_sh, x_cv_sh, y_cv_sh,
                #learning_rate=0.003, reg_term=0.03, 95%
                learning_rate=0.003, reg_term=0.03,
                batch_size=100, n_epochs=32,
                max_its=20000, improv_thresh=0.01, max_its_incr=4,
                x=x,
                rel_val_tol=4e-3,
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

        print("saving model...")
        with open("cnn_model.pkl", "wb") as f:
            pickle.dump(clf, f)

    acc = theano.function([clf.inp, y], clf.score(y))
    te_len = x_te.shape[0]
    print("accuracy: %.2f%%" % (100*acc(
        np.reshape(x_te, (te_len, 1, 28, 28)),
        #np.reshape(x_te, (te_len, 1, 48, 32)),
        y_te)))

def norm(arr):
    minn = arr.min()
    maxx = arr.max()
    rng = maxx - minn
    return (arr - minn)/rng

def visualize():
    import cv2
    import pylab

    print("loading model...", flush=True)
    with open("cnn_model.pkl", "rb") as f:
        clf = pickle.load(f)

    for l in range(2):
        print("layer", l)
        layer = clf.conv_pool_layers[l]["conv"].params
        kernels = layer[0].get_value(borrow=True)
        n_out, n_in, *shape = kernels.shape
        print("kernels shape:", kernels.shape)
        for i in range(n_in):
            print("inp:", i+1)
            for j in range(min(n_out, 8)):
                print("\tin", j+1)
                k = norm(kernels[j][i])
                #print(k)
                pylab.subplot(2, 4, j+1)
                pylab.axis("off")
                pylab.imshow(k)
                pylab.gray();
            pylab.show()
    # recall that the convOp output (filtered image) is actually a "minibatch",
    # of size 1 here, so we take index 0 in the first dimension:
    #pylab.subplot(1, 3, 2)
    #pylab.axis('off')
    #pylab.imshow(filtered_img[0, 0, :, :])
    #pylab.subplot(1, 3, 3)
    #pylab.axis('off')
    #pylab.imshow(filtered_img[0, 1, :, :])

if __name__ == "__main__":
    visualize()
    #main()
