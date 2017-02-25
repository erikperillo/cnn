#!/usr/bin/env python3

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

    clf = mlp.MultiLayerPerceptron(x,
        n_in=x_tr.shape[1], n_hidden=64, n_out=10)

    acc = theano.function([x, y], clf.score(y))
    with_validation = True
    x_tr = np.array(x_tr, dtype="float64")
    y_tr = np.array(y_tr, dtype="int32")
    if with_validation:
        print("calling sgd_with_validation", flush=True)
        x_cv = np.array(x_cv, dtype="float64")
        x_te = np.array(x_te, dtype="float64")
        y_cv = np.array(y_cv, dtype="int32")
        y_te = np.array(y_te, dtype="int32")
        sgd.sgd_with_validation(clf,
            x_tr, y_tr, x_cv, y_cv,
            learning_rate=0.01, reg_term=0.00005,
            batch_size=256, n_epochs=1000,
            max_its=5000, improv_thresh=0.01, max_its_incr=4,
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
