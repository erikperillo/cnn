from theano.tensor.signal import pool
import theano
import theano.tensor as T
import numpy

input = T.dtensor4('input')
maxpool_shape = (4, 4)
pool_out = pool.pool_2d(input, maxpool_shape, ignore_border=True)
f = theano.function([input],pool_out)

invals = numpy.random.RandomState(1).rand(3, 2, 10, 10)
print('With ignore_border set to True:')
print('invals[0, 0, :, :] =\n', invals[0, 0, :, :])
print('output[0, 0, :, :] =\n', f(invals)[0, 0, :, :])

pool_out = pool.pool_2d(input, maxpool_shape, ignore_border=False)
f = theano.function([input],pool_out)
print('With ignore_border set to False:')
print('invals[1, 0, :, :] =\n ', invals[1, 0, :, :])
print('output[1, 0, :, :] =\n ', f(invals)[1, 0, :, :])
