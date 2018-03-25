import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops


def define_forward_pass(X, init_parameters, n_inputs, HL1N, HL2N, sigma=1.0,
                        keep_sparse=True, n_layers=2):
    """
    Defines a Multilayer Perceptron (MLP) model.
    Inputs:
        - X: tf placeholder [batchsize, n_inputs] for feature inputs
        - init_parameters: tuple of numpy arrays (or None), holding values to initialise
                the network weight matrices (and biases). If None, random values are
                used for network initialisation.
        - n_inputs: number of features
        - HL1N: hidden layer 1 size
        - Hl2N: hidden layer 2 size
        - sigma: Variance of Gaussian distribution used when randomly initalising weights
        - keep_sparse: whether to enforce network weight sparsity
        - n_layers: number of hidden layers. Default case is 2, but can also run with 1 or 3,
            but the number of neurons is then defined by HL1N, HL2N.
            Note: The RF initialisation can only be used with n_layers = 2.
    Outputs:
        - predictions: tf tensor holding model predictions
    """

    if n_layers == 2:   # default case.

        # set the inital network parameter values, either to random, ...
        if init_parameters is None:
            # random initial parameters
            W_01 = tf.Variable(tf.random_normal([n_inputs, HL1N], 0.0, sigma))
            W_12 = tf.Variable(tf.random_normal([HL1N, HL2N], 0.0, sigma))
            W_23 = tf.Variable(tf.random_normal([HL2N, 1], 0.0, sigma))
            b_1 = tf.Variable(tf.random_normal([HL1N], 0.0, sigma))
            b_2 = tf.Variable(tf.random_normal([HL2N], 0.0, sigma))
            b_3 = tf.Variable(tf.random_normal([1], 0.0, sigma))

        else:
            # ... or to the specific values induced by the Random-Forest
            W1, b1, W2, b2, W3 = init_parameters

            W_01 = tf.Variable(W1)
            W_12 = tf.Variable(W2)
            W_23 = tf.Variable(W3)
            b_1 = tf.Variable(b1)
            b_2 = tf.Variable(b2)
            b_3 = tf.Variable( np.sum(W3) )

            if keep_sparse:
                mask1 = tf.constant(np.float32(W1!=0.0))
                mask2 = tf.constant(np.float32(W2!=0.0))
                W_01 = tf.multiply(mask1, W_01)
                W_12 = tf.multiply(mask2, W_12)

        # defining the network with the given weights/biases
        h = tf.nn.tanh(tf.matmul(X, W_01) + b_1)
        h2 = tf.nn.tanh(tf.matmul(h, W_12) + b_2 )
        prediction = tf.matmul(h2, W_23) + b_3


    elif n_layers == 1:   # standard 1-layer MLP, initialised randomly
        W_01 = tf.Variable(tf.random_normal([n_inputs, HL1N], 0.0, sigma))
        W_12 = tf.Variable(tf.random_normal([HL1N, 1], 0.0, sigma))
        b_1 = tf.Variable(tf.random_normal([HL1N], 0.0, sigma))
        b_2 = tf.Variable(tf.random_normal([1], 0.0, sigma))
        h = tf.nn.tanh(tf.matmul(X, W_01) + b_1)
        prediction = tf.matmul(h, W_12) + b_2

    elif n_layers == 3:   # standard 3-layer MLP, initialised randomly
        W_01 = tf.Variable(tf.random_normal([n_inputs, HL1N], 0.0, sigma))
        W_12 = tf.Variable(tf.random_normal([HL1N, HL2N], 0.0, sigma))
        W_23 = tf.Variable(tf.random_normal([HL2N, HL2N], 0.0, sigma))
        W_34 = tf.Variable(tf.random_normal([HL2N, 1], 0.0, sigma))
        b_1 = tf.Variable(tf.random_normal([HL1N], 0.0, sigma))
        b_2 = tf.Variable(tf.random_normal([HL2N], 0.0, sigma))
        b_3 = tf.Variable(tf.random_normal([HL2N], 0.0, sigma))
        b_4 = tf.Variable(tf.random_normal([1], 0.0, sigma))

        h = tf.nn.tanh(tf.matmul(X, W_01) + b_1)
        h2 = tf.nn.tanh(tf.matmul(h, W_12) + b_2 )
        h3 = tf.nn.tanh(tf.matmul(h2, W_23) + b_3 )
        prediction = tf.matmul(h3, W_34) + b_4

    return prediction




def run_neural_net(data, init_parameters=None, HL1N=20, HL2N=10, n_layers=2,
                   verbose=True, learning_rate=0.001, forest=None, keep_sparse=True,
                   batchsize=32, n_iterations=100):
    """
    Trains / evaluates a Multilayer perceptron (MLP), potentially with a prespecified
    weight matrix initialisation.
    Inputs:
    - data: tuple of input (X) - output (Y) data for train/dev/test set.
        Output of dataloader.split_data
    - init_parameters: output of initialiser.get_network_initialisation_parameters.
        if init_parameters is set to None, random initial weights are picked.
    - HL1N: number of neurons in first hidden layer
    - HL2N: number of neurons in second hidden layer
    - n_layers: number of hidden layers. Default 2, but can also be used with 1 and 3.
    - verbose: how much to print
    - learning_rate: used during training
    - forest: a pre-trained random forest model. Not relevant when random initialisation is used.
    - keep_sparse: whether to enforce weight matrix sparsity during training
    - batchsize: used during training
    - n_iterations: Number of training epochs
    """

    if verbose:
        print("training MLP...")
    XTrain, XValid, XTest, YTrain, YValid, YTest = data
    n_samples, n_inputs = XTrain.shape
    batchsize = min(batchsize, n_samples)
    ops.reset_default_graph()


    # placeholders
    X = tf.placeholder("float", [None, n_inputs])
    Y = tf.placeholder("float", [None, 1])

    # forward pass
    prediction = define_forward_pass(X, init_parameters, n_inputs, HL1N, HL2N, n_layers=n_layers)

    # defining a RMSE objective function
    loss = tf.reduce_mean(tf.pow(prediction - Y, 2) )
    optimiser = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # define minibatch boundaries
    batch_boundaries = list(zip(range(0, n_samples, batchsize), \
                range(batchsize, n_samples, batchsize)))
    if n_samples % batchsize:
        batch_boundaries += [(batch_boundaries[-1][1],n_samples)]
    if len(batch_boundaries) == 0:
        batch_boundaries += [(0,n_samples)]


    RMSE_train, RMSE_valid, RMSE_test = [], [], []
    pred_test_store = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(n_iterations):
            #shuffle training data new in every epoch
            perm = np.random.permutation(n_samples)
            XTrain = XTrain[perm,:]
            YTrain = YTrain[perm]


            for start, end in batch_boundaries:
                #feed in training data minibatch-wise
                sess.run(optimiser, feed_dict = {X: XTrain[start:end], \
                                                Y: YTrain[start:end]})

            pred_train = sess.run(prediction, feed_dict={X: XTrain, Y: YTrain})
            pred_valid = sess.run(prediction, feed_dict={X: XValid, Y: YValid})
            pred_test = sess.run(prediction, feed_dict={X: XTest, Y: YTest})
            pred_test_store.append(pred_test)

            diff_train = YTrain - pred_train
            RMSE_train.append( np.sqrt(np.mean(np.square(diff_train ) ) ) )

            diff_valid = YValid - pred_valid
            RMSE_valid.append( np.sqrt(np.mean(np.square(diff_valid ) ) ) )

            diff_test = YTest - pred_test
            RMSE_test.append( np.sqrt(np.mean(np.square(diff_test ) ) ) )
            if verbose:
                printstring = "Epoch: {}, Train/Valid RMSE: {}"\
                        .format(i, np.array([RMSE_train[-1], RMSE_valid[-1]]))
                print (printstring)


    # minimum validation error
    amin = np.argmin(RMSE_valid)
    if verbose:
        print ("argmin at", amin )
        print ("valid:", RMSE_valid[amin] )
        print ("test:", RMSE_test[amin] )



    if forest is None:  # vanilla neural net
        return RMSE_test[amin], pred_test_store[amin]
    else:               # RF-initialised neural net

        # In some cases, the tuned RF is not better than the original RF.
        # Validation accuracy is used to identify these cases.

        # compute RF validation performance
        RF_predictions_valid = forest.predict(XValid)
        RF_score_valid = np.sqrt( np.mean (np.square(RF_predictions_valid-np.squeeze(YValid) ) )  )

        # if RF validation performance is better than for neural model
        if RF_score_valid < RMSE_valid[amin]:
            # Case Yes -- return forest score / predictions
            RF_predictions_test = forest.predict(XTest)
            RF_score_test = np.sqrt( np.mean (np.square(RF_predictions_test-np.squeeze(YTest) ) )  )
            return RF_score_test, RF_predictions_test
        else:
            # Case No -- return tuned model score / predictions
            return RMSE_test[amin], pred_test_store[amin]
