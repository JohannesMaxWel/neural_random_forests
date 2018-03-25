import numpy as np

def load_data(dataset_name:str, seed=42):
    """
    Loads a dataset and returns it split into train/valid/test parts.
    """
    if dataset_name == 'mpg':
        import preprocessing.preprocess_mpg as mpg #Miles per Gallon dataset
        X,Y = mpg.load_data()
    elif dataset_name == "boston":
        import preprocessing.preprocess_boston as boston
        X,Y = boston.load_data()
    elif dataset_name == "crimes":
        import preprocessing.preprocess_crimes as crimes
        X,Y = crimes.load_data()
    elif dataset_name == "fires":
        import preprocessing.preprocess_fires as fires
        X,Y = fires.load_data()
    elif dataset_name == "wisconsin":
        import preprocessing.preprocess_wisconsin as wisconsin
        X,Y = wisconsin.load_data()
    elif dataset_name == "concrete":
        import preprocessing.preprocess_concrete as concrete
        X,Y = concrete.load_data()
    elif dataset_name == "protein":
        import preprocessing.preprocess_protein as protein
        X,Y = protein.load_data()
    else:
        raise NameError("Dataset name not recognised: " + dataset_name)
    return split_data(X,Y,seed)


def split_data(X, Y, seed):
    """
    This function takes the data of a supervised learning task and splits it
    into training (50%), validation (25%) and test (25%) portions, which it returns.
    X: model input data (features). numpy array with shape [n_samples, n_features]
    Y: model output data (labels). numpy array with shape [n_samples]
    seed: random seed
    """
    np.random.seed(seed)
    n_samples = X.shape[0]

    # shuffle data
    permutation = np.random.permutation(n_samples)
    X_perm = X[permutation, :]
    Y_perm = Y[permutation]

    # indices for the positions in the dataset where validation and test data begin
    split1 = int(0.5*n_samples)
    split2 = int(0.75*n_samples)

    # split the permuted inputs into three portions
    X_Train = X_perm[:split1, :]
    X_Valid = X_perm[split1 + 1 : split2, :]
    X_Test = X_perm[split2 + 1 : , :]

    # split the permuted outputs into three portions
    Y_Train = Y_perm[:split1]
    Y_Valid = Y_perm[split1 + 1 : split2]
    Y_Test = Y_perm[split2 + 1 :]

    return X_Train, X_Valid, X_Test, Y_Train, Y_Valid, Y_Test



def dump2csv(dataset_name:str):
    """
    Loads a dataset and dumps it in csv format
    """
    if dataset_name == 'mpg':
        import preprocessing.preprocess_mpg as mpg #Miles per Gallon dataset
        X,Y = mpg.load_data()
    elif dataset_name == "boston":
        import preprocessing.preprocess_boston as boston
        X,Y = boston.load_data()
    elif dataset_name == "crimes":
        import preprocessing.preprocess_crimes as crimes
        X,Y = crimes.load_data()
    elif dataset_name == "fires":
        import preprocessing.preprocess_fires as fires
        X,Y = fires.load_data()
    elif dataset_name == "wisconsin":
        import preprocessing.preprocess_wisconsin as wisconsin
        X,Y = wisconsin.load_data()
    elif dataset_name == "concrete":
        import preprocessing.preprocess_concrete as concrete
        X,Y = concrete.load_data()
    elif dataset_name == "protein":
        import preprocessing.preprocess_protein as protein
        X,Y = protein.load_data()
    else:
        raise NameError("Dataset name not recognised: " + dataset_name)

    out_dir = 'datasets/csv_data/{}/'.format(dataset_name)

    np.savetxt(out_dir+'X.csv', X, delimiter=",")
    np.savetxt(out_dir+'Y.csv', np.squeeze(Y), delimiter=",")


if __name__ == "__main__":
    dump2csv('mpg')
