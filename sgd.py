import theano
from theano import tensor
import timeit
import numpy as np

def sgd(est, 
        x_data, y_data,
        learning_rate=0.1, reg_term=0.1,
        batch_size=1, n_epochs=None,
        max_its=1e6, rel_tol=1e-3,
        verbose=True, print_flush_period=100):
    """
    Stochastic Gradient Descent with regularization.
    Parameters:
    *est: Estimator object. It must have the following attributes:
        .x: input
        .y: output
        .cost: cost function (without regularization term)
        .reg: regularization term of cost function
        .grad: list of tuples in format
            (param, g_cost_wrt_param, g_reg_wrt_param)
    *x_data: x input matrix.
    *y_data: y output vector.
    *learning_rate: Learning rate.
    *reg_term: Regularization term.
    *batch_size: Samples for each batch. If None, batch_size = n_samples.
    *n_epochs: Number of epochs to run.
        If None, runs until another stopping criteria.
    *max_its: Maximum number of iterations to run.
    *rel_tol: Relative difference of current/last batch cost. Not used if None.
    *verbose: Print info if True, print nothing otherwise.
    *print_flush_period: Period to flush printed information.
    """

    #info function, only prints something if verbose is True
    info = print if verbose else lambda *args, **kwargs: None

    #setting batch size
    n_samples = x_data.shape[0]
    if batch_size is None:
        batch_size = n_samples

    #variable for indexing training samples
    index = tensor.lscalar(name="index")

    #shared objects
    x_data_shared = theano.shared(np.asarray(x_data, dtype=x_data.dtype),
        borrow=True)
    y_data_shared = theano.shared(np.asarray(y_data, dtype=y_data.dtype),
        borrow=True)

    #updates variables
    updates = [(var, var - learning_rate*(d_cost_var + reg_term*d_reg_var)) \
        for var, d_cost_var, d_reg_var in est.grad]

    #compiling function
    info("making func...", end="", flush=True)
    train_f = theano.function(
        inputs=[index],
        outputs=est.cost,
        updates=updates,
        givens={
            est.x: x_data_shared[index*batch_size:(index+1)*batch_size],
            est.y: y_data_shared[index*batch_size:(index+1)*batch_size]
        })
    info(" done")

    start_time = timeit.default_timer()
    epoch_count = 0
    it_count = 0
    done_looping = False
    n_batches = n_samples//batch_size
    last_batch_cost = None

    #main loop
    info("starting optimization")
    while True:
        if done_looping:
            break

        #epochs stopping criteria
        if n_epochs and epoch_count >= n_epochs:
            info("WARNING: reached number of epochs")
            done_looping = True
            continue

        #iterating over batches
        for batch_idx in range(n_batches):
            batch_cost = train_f(batch_idx)

            info("[iter %d][epoch %d] batch_cost = %f" %\
                (it_count, epoch_count, batch_cost), end="")

            #relative cost stopping criteria
            if last_batch_cost and rel_tol:
                rel_batch_cost = abs(1 - last_batch_cost/batch_cost)
                info(" | rel_batch_cost = %f" % rel_batch_cost)
                if rel_batch_cost <= rel_tol:
                    info("rel_tol criteria matched")
                    done_looping = True
                    break
            else:
                info("")

            info(end="", flush=it_count%print_flush_period == 0)

            #iterations number stopping criteria
            it_count += 1
            if max_its and it_count >= max_its:
                info("WARNING: maximum iterations reached")
                done_looping = True
                break

            last_batch_cost = batch_cost

        epoch_count += 1

    elapsed_time = timeit.default_timer() - start_time
    info("elapsed time: %fs" % elapsed_time)
