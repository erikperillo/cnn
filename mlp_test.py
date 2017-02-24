#!/usr/bin/env python3

import mlp
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

    clf = mlp.MultiLayerPerceptron(x, y,
        n_in=x_data.shape[1], n_hidden=8, n_out=2)

    print("calling sgd")
    sgd.sgd(clf, x_data, y_data,
        learning_rate=0.1,
        reg_term=1,
        batch_size=None,
        rel_tol=2e-3,
        n_epochs=128,
        verbose=True)

    acc = theano.function([x, y], clf.score)
    print("accuracy: %.2f%%" % (100*acc(x_data, y_data)))

if __name__ == "__main__":
    main()
