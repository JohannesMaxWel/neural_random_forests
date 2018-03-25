import numpy as np
from tqdm import tqdm
import tensorflow
from forest_fitting import fit_random_forest
from initialiser import get_network_initialisation_parameters
from feedforward import run_neural_net

from tensorflow.python.framework import ops


def individually_trained_networks(data, ntrees, depth, keep_sparse=False, verbose=True):
    """
    Method 1. Trains ntree single MLPs independently, each with the initialisation
    of a single tree from a random forest.
    The predictions of the resulting models are ensembled.
    Inputs:
        - data: a tuple of XTrain, XValid, XTest, YTrain, YValid, YTest, corresponding
            to outputs of dataloader.split_data
        - ntrees: int. number of trees
        - depth: int. maximum tree depth
        - keep_sparse: bool. whether to enforce sparsity of the NN weights during training.
        - verbose: bool. more/less printouts
    Output:
        - Test set predictions (np array)
        - Test set Root Mean Square Error (RMSE)
    """

    # the dataset, with inputs (X), and outputs (Y)
    XTrain, XValid, XTest, YTrain, YValid, YTest = data

    # keep track for each individual tree in the forest
    individual_network_errors, individual_network_predictions = [], []

    print("Tuning MLPs individually for a total of {} trees. Weight sparsity enforced: {}".format(str(ntrees), str(keep_sparse)))
    for i in tqdm(range(ntrees)):   # loop over trees

        if verbose:
            print ("Training network for tree", i+1, "of", ntrees )

        # new random state
        random_state = np.random.randint(0,100000,1)[0]
        ops.reset_default_graph()

        # train RF with single tree
        rf, rf_results = fit_random_forest(data, 1, depth, random_state, verbose=False)

        # single tree: generate predictions
        yhatRF_test = rf.predict(XTest)
        yhatRF_valid = rf.predict(XValid)

        # evaluate performance (RMSE) of single tree
        diff_t = yhatRF_test-np.squeeze(YTest)
        diff_v = yhatRF_valid-np.squeeze(YValid)
        RF_score_t = np.sqrt( np.mean (np.square(diff_t) )  )
        RF_score_v = np.sqrt( np.mean (np.square(diff_v) )  )

        # extract network initialisation parameters for network based on one tree
        init_parameters = get_network_initialisation_parameters(rf)

        # train network
        RMSE, pred = run_neural_net(data, init_parameters,
                            verbose=False, forest=rf, keep_sparse=keep_sparse)

        individual_network_errors += [RMSE]
        individual_network_predictions += [np.atleast_2d(np.ravel(pred))]

    # average the (scalar) predictions across all trees
    all_preds = np.concatenate(individual_network_predictions, axis=0)
    avg_pred = np.mean( all_preds, axis=0)

    # compute RMSE
    RMSE = np.sqrt( np.mean (np.square(avg_pred - np.squeeze(YTest)) )  )
    if verbose:
        print ("score for averaged prediction of individual networks:", RMSE )

    return RMSE, avg_pred
