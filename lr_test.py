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

    clf = lr.LogisticRegression(x, y, x_data.shape[1], 2)

    sgd.sgd(clf, x_data, y_data,
        learning_rate=0.1,
        reg_term=1,
        batch_size=10,
        rel_tol=None,
        n_epochs=128,
        verbose=True)

    print((y_data.dtype))
    s = clf.score
    f = theano.function([x, y], s)
    print(f(x_data, y_data))

if __name__ == "__main__":
    main()
