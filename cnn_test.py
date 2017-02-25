#!/usr/bin/env python3

import pandas as pd
import numpy as np
import theano
from theano import tensor
import sgd
import cnn
import mlp
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

    #BUILDING LENET
    x = tensor.matrix("x")
    y = tensor.ivector("y")

    batch_size = 32
    imgs_h, imgs_w = 28, 28

    #input
    cnn_input = x.reshape((batch_size, 1, imgs_h, imgs_w))

    #layer 0:
    n_layer_0_out_maps = 3
    #conv layer
    layer_0_conv = cnn.ConvolutionLayer(
        inp=layer_0_input,
        n_in_maps=1,
        n_out_maps=n_layer_0_out_maps,
        filter_shape=(5, 5),
        activation_f=tensor.nnet.sigmoid,
        w_init_f="normal")
    #pooling layer
    layer_0_pool = cnn.PoolingLayer(
        inp=layer_0_conv.output,
        shape=(2, 2),
        pool_f="max",
        ignore_border=True)

    #layer 1:
    n_layer_1_out_maps = 9
    #conv layer
    layer_1_conv = cnn.ConvolutionLayer(
        inp=layer_0_pool.output,
        n_in_maps=n_layer_0_out_maps,
        n_out_maps=n_layer_1_out_maps,
        filter_shape=(5, 5),
        activation_f=tensor.nnet.sigmoid,
        w_init_f="normal")
    #pooling layer
    layer_1_pool = cnn.PoolingLayer(
        inp=layer_1_conv.output,
        shape=(2, 2),
        pool_f="max",
        ignore_border=True)

    #layer 2 (fully connected):
    layer_2_input = layer_1_pool.output.flatten(2)
    cnn_output = tensor.ivector("cnn_output")
    layer_2_mlp = mlp.MultiLayerPerceptron(
        x=layer_2_input, y=cnn_output,
        n_in=n_layer_1_out_maps*4*4,
        n_hidden=64,
        n_out=10,
        activation_f=tensor.tanh,
        w_init_f="xavier_tanh",
        reg="l2")

if __name__ == "__main__":
    main()
