import numpy as np
from sklearn.ensemble import RandomForestRegressor


def fit_random_forest(data, ntrees=30, depth=6, random_state=42, verbose=False):
    """
    Fits a random regression forest to some data and returns the model.
    """
    if verbose:
        print('Fitting Random Forest...')

    rf = RandomForestRegressor(ntrees,
                               'mse',
                               depth,
                               min_samples_split=2,
                               min_samples_leaf=1,
                               min_weight_fraction_leaf=0.0,
                               max_features='auto',
                               max_leaf_nodes=None,
                               bootstrap=False, oob_score=False,
                               n_jobs=1,
                               random_state=random_state,
                               verbose=0,
                               warm_start=False)

    # fit the forest
    XTrain, XValid, XTest, YTrain, YValid, YTest = data
    rf.fit(XTrain, np.ravel(YTrain))

    # generate predictions
    RF_predictions_train = rf.predict(XTrain)
    RF_predictions_valid = rf.predict(XValid)
    RF_predictions_test = rf.predict(XTest)

    # compute RMSE metrics for predictions
    RF_score_train = np.sqrt( np.mean (np.square(RF_predictions_train-np.squeeze(YTrain) ) )  )
    RF_score_valid = np.sqrt( np.mean (np.square(RF_predictions_valid-np.squeeze(YValid) ) )  )
    RF_score_test = np.sqrt( np.mean (np.square(RF_predictions_test-np.squeeze(YTest) ) )  )
    if verbose:
        print ("RF score (RMSE) train: ", RF_score_train)
        print ("RF score (RMSE) valid: ", RF_score_valid)
        print ("RF score (RMSE) test: ", RF_score_test)
    rf_results = (RF_score_train, RF_score_valid, RF_score_test)

    return rf, rf_results
