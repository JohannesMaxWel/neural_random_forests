import numpy as np
import sys
from dataloader import load_data
from forest_fitting import fit_random_forest
from feedforward import run_neural_net
from initialiser import get_network_initialisation_parameters
from individually_trained import individually_trained_networks

# set seed and printoptions.
np.random.seed(42)
np.set_printoptions(precision=4, suppress=True)


def neural_random_forest(dataset_name="mpg"):
    """
    Takes a regression dataset name, and trains/evaluates 4 classifiers:
    - a random forest
    - a 2-layer MLP
    - a neural random forest (method 1)
    - a neural random forest (method 2)
    """
    # pick a regression dataset
    dataset_names = ["boston", "concrete", "crimes", "fires", "mpg", "wisconsin", "protein"]
    if not dataset_name or dataset_name not in dataset_names:
        dataset_name = "mpg"  # set as default dataset

    # load the dataset, with randomised train/dev/test split
    data = load_data(dataset_name, seed=np.random.randint(0,100000,10)[0])

    # X: regression input variable matrix, size [n_data_points, n_features]
    # Y: regression output vector, size [n_data_points]
    # General format of data: 6-tuple
    # XTrain, XValid, XTest, YTrain, YValid, YTest

    # forest hyperparameters
    ntrees = 30
    depth = 6

    # train a random regression forest model
    rf, rf_results = fit_random_forest(data, ntrees, depth, verbose=False)

    # derive initial neural network parameters from the trained rf model
    init_parameters = get_network_initialisation_parameters(rf)

    # determine layer size for layers 1 and 2 in the 2-layer MLP
    HL1N, HL2N = init_parameters[2].shape

    # train a standard 2-layer MLP with HL1N / HL2N hidden neurons in layer 1 / 2.
    NN2,_ = run_neural_net(data, init_parameters=None, HL1N=HL1N, HL2N=HL2N, verbose=False)

    # train many small networks individually, initial weights from a decision tree (method 1)
    method1_full,_  = individually_trained_networks(data, ntrees, depth, keep_sparse=False, verbose=False)
    method1_sparse,_ = individually_trained_networks(data, ntrees, depth, keep_sparse=True, verbose=False)

    # train one large network with sparse initial weights from random forest parameters (method 2)
    method2_full,_ = run_neural_net(data, init_parameters, verbose=False, forest=rf, keep_sparse=False)
    method2_sparse,_ = run_neural_net(data, init_parameters, verbose=False, forest=rf, keep_sparse=True)

    results = {
        "RF": rf_results[2],
        "NN2": NN2,
        "NRF1 full": method1_full,
        "NRF1 sparse": method1_sparse,
        "NRF2 full": method2_full,
        "NRF2 sparse": method2_sparse
        }

    print("RMSE:", results)
    return results



if __name__ == "__main__":
    args = sys.argv
    if len(args) > 1:
        dataset = args[1]
    else:
        dataset = "mpg"
    _ = neural_random_forest(dataset)
