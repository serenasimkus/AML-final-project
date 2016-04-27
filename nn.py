# To run, install lasagne and theano with pip (pip install theano; pip install lasagne)
# Currently a very basic neural net with one hidden layer of 40 nodes, and classic SGD.
# Using cross-entropy error since that's the standard for classification. Unfortunately
# it doesn't perform all that well--gets roughly 50% classified correctly and 92-93%
# classified within one, compared to lasso and even classic ridge regression which get
# > 60% classified correctly and 94% classified correctly within one.
# Make sure to load in data before running:
# python nn_loaddata.py
# To run:
# python nn.py

# TODO: Still need to make some plots for this somehow.

import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import theano
import theano.tensor as T
import numpy as np

import lasagne

def build_nn(input_var=None):
    # build input layer of unspecified batch size of data pts as row vectors
    l_in = lasagne.layers.InputLayer(shape=(None,80), input_var=input_var)

    # first hidden layer, trying 40 units
    l_1 = lasagne.layers.DenseLayer(l_in, num_units=40, nonlinearity=lasagne.nonlinearities.tanh, W=lasagne.init.GlorotUniform())

    # classify into 7 categories
    l_out = lasagne.layers.DenseLayer(l_1, num_units=7, nonlinearity=lasagne.nonlinearities.softmax)

    return l_out

# for batch SGD
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def main(num_epochs=500):
    # X_train, Y_train, X_test, Y_test = load_dataset()
    X_train = np.load('train_data.npy')
    X_test = np.load('test_data.npy')
    Y_train = np.load('train_targets.npy')
    Y_test = np.load('test_targets.npy')

    input_var = T.matrix('inputs', dtype='float64')
    target_var = T.ivector('targets')

    network = build_nn(input_var)
    pred = lasagne.layers.get_output(network)
    # cross-entropy loss
    loss = lasagne.objectives.categorical_crossentropy(pred, target_var)
    # squared error loss
    # loss = lasagne.objectives.squared_error(T.argmax(pred, axis=1), target_var)
    loss = loss.mean()

    # Perform SGD with momentum
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.sgd(loss, params, learning_rate=0.1)

    # for testing:
    test_pred = lasagne.layers.get_output(network)
    # cross-entropy loss
    test_loss = lasagne.objectives.categorical_crossentropy(test_pred, target_var)
    # squared error loss
    # test_loss = lasagne.objectives.squared_error(T.argmax(test_pred, axis=1), target_var)
    test_loss = test_loss.mean()

    # percentage of correctly classified
    test_acc = T.mean(T.eq(T.argmax(test_pred, axis=1), target_var), dtype=theano.config.floatX)

    # percentage within one
    within_one = T.mean(T.and_(T.le(T.argmax(test_pred, axis=1), target_var + 1), T.ge(T.argmax(test_pred, axis=1), target_var - 1)), dtype=theano.config.floatX)

    # diff
    each_diff = T.abs_(T.argmax(test_pred, axis=1) - target_var)

    # training function
    train_fn = theano.function([input_var, target_var], loss, updates=updates, allow_input_downcast=True)

    # testing function
    test_fn = theano.function([input_var, target_var], [test_loss, test_acc, within_one, each_diff], allow_input_downcast=True)

    all_train_err = []
    # Train
    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, Y_train, 100, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        all_train_err.append(train_err)
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))

    # Test
    test_err = 0
    test_acc = 0
    within_one = 0
    test_batches = 0
    diff = None
    for batch in iterate_minibatches(X_test, Y_test, 100, shuffle=False):
        inputs, targets = batch
        err, acc, one, each_diff = test_fn(inputs, targets)
        if diff is None:
            diff = each_diff
        else:
            diff = T.concatenate([diff, each_diff])
        test_err += err
        test_acc += acc
        within_one += one
        test_batches += 1
    print("Test results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))
    print("  test accuracy within one:\t\t{:.2f} %".format(
        within_one / test_batches * 100))

    # plot training error
    plt.figure(1)
    plt.plot(all_train_err)
    plt.ylabel('error')
    plt.xlabel('epochs')

    # plot predictions errs
    plt.figure(2)
    plt.plot(diff.eval(), 'o')
    plt.ylabel('absolute difference')
    plt.show()
    np.savez('nn_model.npz', lasagne.layers.get_all_param_values(network))

main(500)
