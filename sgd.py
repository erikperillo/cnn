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
    convergence_crit = None

    #main loop
    info("starting optimization")
    while not done_looping:
        #epochs stopping criteria
        if n_epochs and epoch_count >= n_epochs:
            info("WARNING: reached number of epochs")
            done_looping = True
            convergence_crit = "n_epochs"
            continue

        #iterating over batches
        for batch_idx in range(n_tbatches):
            batch_cost = train_f(batch_idx)

            info("[iter %d][epoch %d][batch %d/%d] batch_cost = %f" %\
                (it_count, epoch_count, batch_idx, n_batches, batch_cost),
                end="")

            #relative cost stopping criteria
            if last_batch_cost and rel_tol:
                rel_batch_cost = abs(1 - last_batch_cost/batch_cost)
                info(" | rel_batch_cost = %f" % rel_batch_cost, end="")
                if rel_batch_cost <= rel_tol:
                    info("\nWARNING: rel_tol criteria matched")
                    done_looping = True
                    convergence_crit = "rel_tol"
                    break

            info("", flush=it_count%print_flush_period == 0)

            #iterations number stopping criteria
            it_count += 1
            if max_its and it_count >= max_its:
                info("WARNING: maximum iterations reached")
                done_looping = True
                convergence_crit = "max_its"
                break

            last_batch_cost = batch_cost

        epoch_count += 1

    elapsed_time = timeit.default_timer() - start_time
    info("[end of sgd] elapsed time: %fs\n" % elapsed_time)

    return convergence_crit

def sgd_with_validation(est, 
        x_train_data, y_train_data,
        x_val_data, y_val_data,
        learning_rate=0.1, reg_term=0.1,
        batch_size=1, n_epochs=None,
        max_its=5000, improv_thresh=0.01, max_its_incr=4,
        rel_val_tol=1e-3,
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
        .score: accuracy (in [0, 1])
    *x_data: x input matrix.
    *y_data: y output vector.
    *learning_rate: Learning rate.
    *reg_term: Regularization term.
    *batch_size: Samples for each batch. If None, batch_size = n_samples.
    *n_epochs: Number of epochs to run.
        If None, runs until another stopping criteria.
    *max_its: Maximum number of iterations to run.
    *max_its_incr: Maximum iterations to increase when a new best is found.
    *verbose: Print info if True, print nothing otherwise.
    *print_flush_period: Period to flush printed information.
    """

    #info function, only prints something if verbose is True
    info = print if verbose else lambda *args, **kwargs: None

    #setting batch size
    n_train_samples = x_train_data.shape[0]
    n_val_samples = x_val_data.shape[0]
    if batch_size is None:
        batch_size = n_train_samples

    #variable for indexing training samples
    index = tensor.lscalar(name="index")

    #shared objects
    x_train_data_shared = theano.shared(
        np.asarray(x_train_data, dtype=x_train_data.dtype), borrow=True)
    y_train_data_shared = theano.shared(
        np.asarray(y_train_data, dtype=y_train_data.dtype), borrow=True)
    x_val_data_shared = theano.shared(
        np.asarray(x_val_data, dtype=x_val_data.dtype), borrow=True)
    y_val_data_shared = theano.shared(
        np.asarray(y_val_data, dtype=y_val_data.dtype), borrow=True)

    #updates variables
    updates = [(var, var - learning_rate*(d_cost_var + reg_term*d_reg_var)) \
        for var, d_cost_var, d_reg_var in est.grad]

    #compiling functions
    info("making train/val func...", end="", flush=True)
    train_f = theano.function(
        inputs=[index],
        outputs=est.cost,
        updates=updates,
        givens={
            est.x: x_train_data_shared[index*batch_size:(index+1)*batch_size],
            est.y: y_train_data_shared[index*batch_size:(index+1)*batch_size]
        })
    val_f = theano.function(
        inputs=[],
        outputs=est.score,
        updates=updates,
        givens={
            est.x: x_val_data_shared,#[index*batch_size:(index+1)*batch_size],
            est.y: y_val_data_shared#[index*batch_size:(index+1)*batch_size]
        })
    info(" done")

    #number of epochs
    epoch_count = 0
    #total iterations (over batcher) count
    it_count = 0
    #true if its time to stop
    done_looping = False
    #number of train batches
    n_train_batches = n_train_samples//batch_size
    #number of validation batches
    n_val_batches = max(n_val_samples//batch_size, 1)
    #validation frequency
    val_freq = min(n_train_batches, max_its//2)
    #best validation score
    best_val_score = 0
    #last validation score
    last_val_score = 0
    #starting time
    start_time = timeit.default_timer()
    #convergence criteria
    convergence_crit = None

    #main loop
    info("starting optimization")
    while not done_looping:
        #epochs stopping criteria
        if n_epochs and epoch_count >= n_epochs:
            info("WARNING: reached number of epochs")
            done_looping = True
            convergence_crit = "n_epochs"
            continue

        #iterating over batches
        for batch_idx in range(n_train_batches):
            batch_cost = train_f(batch_idx)

            info("[iter %d][epoch %d][batch %d/%d] batch_cost = %f" %\
                (it_count, epoch_count, batch_idx, n_train_batches, batch_cost),
                end="")
            
            if it_count and it_count%val_freq == 0:
                val_score = val_f()
                info(" | val_score = %f" % val_score, end="")

                if val_score > best_val_score:
                    if val_score >= best_val_score*(1+improv_thresh):
                        max_its = max(max_its, it_count+max_its_incr)

                    best_val_score = val_score
                    info(" (BEST SCORE SO FAR)", end="")

                #relative cost stopping criteria
                if last_val_score and rel_val_tol:
                    rel_val_score = abs(1 - last_val_score/val_score)
                    info(" | rel_val_score = %f" % rel_val_score, end="")
                    if rel_val_score <= rel_val_tol:
                        info("\nWARNING: rel_val_tol criteria matched")
                        done_looping = True
                        convergence_crit = "rel_val_tol"
                        break

                last_val_score = val_score

            info("", flush=it_count%print_flush_period == 0)

            #iterations number stopping criteria
            it_count += 1
            if max_its and it_count >= max_its:
                info("WARNING: maximum iterations reached")
                done_looping = True
                convergence_crit = "max_its"
                break

        epoch_count += 1

    elapsed_time = timeit.default_timer() - start_time
    info("[end of sgd_with_validation] elapsed time: %fs\n" % elapsed_time)

    return convergence_crit
