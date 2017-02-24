import theano
from theano import tensor
import timeit

def sgd(self, x_data, y_data, n_epochs=1000, patience=5000, verbose=True):
    x = tensor.matrix(name="x")
    y = tensor.matrix(name="y")
    index = tensor.lscalar()

    train_f = theano.function(
        inputs=[index],
        outputs=cost_f,
        updates=updates,
        givens={
            x: x_data[index*batch_size:(index+1)*batch_size],
            y: y_data[index*batch_size:(index+1)*batch_size]
        })

    start_time = timeit.default_timer()
    epoch_count = 0
    done_looping = False

    while epoch_count < n_epochs and not done_looping:
        mini_batch_cost = self.train_f



