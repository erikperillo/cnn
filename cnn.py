#!/usr/bin/env python3


from PIL import Image
import pylab
import numpy as np
import theano
from theano import tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d


def imread(filepath):
    img = Image.open(filepath)
    return np.asarray(img, dtype=np.float64) / 255

def get_conv_layer(shape, relu=T.nnet.sigmoid, rand_seed=23455):
    #initializing random weights
    rng = np.random.RandomState(rand_seed)

    #instantiating 4D tensor for input
    inp = T.tensor4(name='inp')

    #initializing shared variable for weights
    assert len(shape) == 4
    w_bound = np.sqrt(np.prod(shape))
    weights = theano.shared(np.asarray(
        rng.uniform(low=-1.0/w_bound, high=1.0/w_bound, size=shape),
        dtype=inp.dtype), name='weights')

    #initializing shared variable for bias (1D tensor) with random values
    bias_shp = (shape[0], )
    bias = theano.shared(np.zeros(shape=bias_shp, dtype=inp.dtype), name='bias')

    #building symbolic expression that computes the convolution of input
    #with filters in w
    conv_out = conv2d(inp, weights)

    #building symbolic expression to add bias and apply activation function,
    #i.e. produce neural net layer output
    relu_out = relu(conv_out + bias.dimshuffle('x', 0, 'x', 'x'))

    #creating theano function to compute filtered images
    f = theano.function([inp], relu_out)

    return f

def get_maxpool_layer(shape=(2, 2), ignore_border=True):
    inp = T.dtensor4('inp')
    pool_out = pool.pool_2d(inp, shape, ignore_border=True)
    f = theano.function([inp], pool_out)

    return f

def conv_layer_conv(f, img):
    # dimensions are (height, width, channel)
    # put image in 4D tensor of shape (1, 3, height, width)
    img_ = img.transpose(2, 0, 1).reshape(1, 3, img.shape[0], img.shape[1])
    filtered_img = f(img_)

    # plot original image and first and second components of output
    pylab.subplot(1, 3, 1)
    pylab.axis('off')
    pylab.imshow(img)
    pylab.gray()

    # recall that the convOp output (filtered image) is actually a "minibatch",
    # of size 1 here, so we take index 0 in the first dimension:
    pylab.subplot(1, 3, 2)
    pylab.axis('off')
    pylab.imshow(filtered_img[0, 0, :, :])
    pylab.subplot(1, 3, 3)
    pylab.axis('off')
    pylab.imshow(filtered_img[0, 1, :, :])
    pylab.show()

def main():
    print("reading image...")
    img = imread("/home/erik/chinchin.jpg")

    print("done")

if __name__ == "__main__":
    main()

