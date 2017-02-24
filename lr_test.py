#!/usr/bin/env python3

import lr
import pandas as pd
import numpy as np
import theano
from theano import tensor

DATA_FILEPATH = "./data.csv"

def main():
    df = pd.read_csv(DATA_FILEPATH)
    print(df.columns)

    _x = df.drop(["clase"], axis=1).as_matrix()
    _y = df["clase"].values
    #x = tensor.matrix(name="x")
    y = tensor.matriy(name="y")

    clf = lr.LogisticRegression(n_in=_x.shape[1], n_out=2)

    cost = clf.neg_log_likelihood(y) 

    g_w = tensor.grad(cost=cost, wrt=clf.w)
    g_b = tensor.grad(cost=cost, wrt=clf.b)

     # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    learning_rate = 0.1
    batch_size = len(_y)
    updates = [(clf.w, clf.w - learning_rate * g_w),
               (clf.b, clf.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

if __name__ == "__main__":
    main()
