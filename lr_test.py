#!/usr/bin/env python3

import lr
import pandas as pd
import numpy as np
import theano
import sgd
from theano import tensor

DATA_FILEPATH = "./data.csv"

def main():
    df = pd.read_csv(DATA_FILEPATH)
    print(df.columns)

    x_data = df.drop(["clase"], axis=1).as_matrix()
    y_data = np.array(list(map(int, df["clase"])), dtype="int32")
    n_samples = x_data.shape[0]
    print(type(y_data))

    x = tensor.matrix(name="x")
    y = tensor.ivector(name="y")

    clf = lr.LogisticRegression(x, x_data.shape[1], 2)

    with_validation = True
    if with_validation:
        val_frac = 0.3
        val_samples = int(n_samples*val_frac)
        train_samples = n_samples - val_samples
        x_tr, y_tr = x_data[:train_samples, :], y_data[:train_samples]
        x_val, y_val = (x_data[train_samples:(train_samples+val_samples), :],
            y_data[train_samples:(train_samples+val_samples)])
        print("calling sgd_with_validation")
        sgd.sgd_with_validation(clf,
            x_tr, y_tr, x_val, y_val,
            learning_rate=0.01, reg_term=0.0001,
            batch_size=32, n_epochs=1000,
            max_its=10000, improv_thresh=0.01, max_its_incr=4,
            rel_val_tol=1e-3,
            verbose=True)
    else:
        print("calling sgd")
        sgd.sgd(clf, 
            x_data, y_data, y=y,
            learning_rate=0.01,
            reg_term=0.0001,
            batch_size=220,
            rel_tol=2e-3,
            n_epochs=256,
            verbose=True)

    acc = theano.function([x, y], clf.score(y))
    print("accuracy: %.2f%%" % (100*acc(x_data, y_data)))

if __name__ == "__main__":
    main()
