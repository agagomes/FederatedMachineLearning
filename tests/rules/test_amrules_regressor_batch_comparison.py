from skmultiflow.rules import AMRulesRegressor, VeryFastDecisionRulesClassifier
from skmultiflow.drift_detection import PageHinkley
from pandas_streaming.df import StreamingDataFrame
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from math import sqrt
from array import array

import time

import logging
logging.basicConfig(filename='logsinfo.log', filemode='w', level=logging.INFO)
LOGGER = logging.getLogger(__name__)


# Data
df = pd.read_csv("/home/angelo/Desktop/databases/cart/CartExample/cart_delve.data",
                          names=list(map(str, range(0,11))), sep="  ")
#df = pd.read_csv("/home/angelo/Desktop/databases/ailerons/Ailerons/ailerons.data",
#                names=list(map(str, range(0,41))))
#df = pd.read_csv("/home/angelo/Desktop/databases/elevators/Elevators/elevators.data",
#                    names=list(map(str, range(0,19))))
#df = pd.read_csv("/home/angelo/Desktop/databases/fried/FriedmanExample/fried_delve.data", 
#                    names=list(map(str, range(0,11))), sep=" ")


def test_amrules_ordered_rules():

    LOGGER.info("\ndf=\n"+str(df.head()))
    
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1:]

    # scaler = StandardScaler().fit(X)
    # X = scaler.transform(X)

    X = X.values.tolist()
    y = y.values.tolist()
    y = [item for sublist in y for item in sublist]
    
    # Model
    # learner_no_order = AMRulesRegressor(
    #                             grace_period=200,
    #                             tie_threshold=0.05,
    #                             expand_confidence=0.0000001,
    #                             rule_prediction='first_hit',#'weighted_max',# 
    #                             ordered_rules=False,
    #                             drift_detector=None,
    #                             remove_poor_atts=True,
    #                             nominal_attributes=None,
    #                             decimal_numbers=4)

    
    learner_order = AMRulesRegressor(
                                grace_period=200,
                                tie_threshold=0.05,
                                expand_confidence=0.0000001,
                                rule_prediction='first_hit',#'weighted_max',# 
                                ordered_rules=True,
                                drift_detector=None,
                                remove_poor_atts=True,
                                nominal_attributes=None,
                                decimal_numbers=4)

    # Train
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    
    scoring = { 'n_mae':'neg_mean_absolute_error', 
                'n_rmse':'neg_root_mean_squared_error' }

    # start = time.time()
    # results = cross_validate(learner_no_order, X, y, cv=kfold, scoring=scoring)
    # end = time.time()
    # LOGGER.info("\nModel: no order\nMAE: %.5f (%.5f), RMSE: %.5f (%.5f)\ntime: %.5f" % 
    #                             (-results['test_n_mae'].mean(), results['test_n_mae'].std(),
    #                             -results['test_n_rmse'].mean(), results['test_n_rmse'].std(),
    #                             (end - start)))

    start = time.time()
    results = cross_validate(learner_order, X, y, cv=kfold, scoring=scoring)
    end = time.time()
    LOGGER.info("\nModel: order\nMAE: %.5f (%.5f), RMSE: %.5f (%.5f)\ntime: %.5f" % 
                            (-results['test_n_mae'].mean(), results['test_n_mae'].std(),
                            -results['test_n_rmse'].mean(), results['test_n_rmse'].std(),
                            (end - start)))


    # Predict
    # y_pred = cross_val_predict(learner_no_drift, X, y, cv=kfold)
   

"""
def test_amrules_drift():

    LOGGER.info("\ndf=\n"+str(df.head()))
    
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1:]

    # scaler = StandardScaler().fit(X)
    # X = scaler.transform(X)

    X = X.values.tolist()
    y = y.values.tolist()
    y = [item for sublist in y for item in sublist]
    
    # Model
    learner_no_drift = AMRulesRegressor(grace_period=200,
                                tie_threshold=0.05,
                                expand_confidence=0.0000001,
                                ordered_rules=True,
                                drift_detector=None,
                                nominal_attributes=None,
                                decimal_numbers=4)

    
    learner_drift = AMRulesRegressor(grace_period=200,
                                tie_threshold=0.05,
                                expand_confidence=0.0000001,
                                ordered_rules=True,
                                drift_detector=PageHinkley(),
                                nominal_attributes=None,
                                decimal_numbers=4)

    
    # Train
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    
    scoring = { 'n_mae':'neg_mean_absolute_error', 
                'n_rmse':'neg_root_mean_squared_error' }

    start = time.time()
    results = cross_validate(learner_no_drift, X, y, cv=kfold, scoring=scoring)
    end = time.time()
    LOGGER.info("\nModel: no drift\nMAE: %.5f (%.5f), RMSE: %.5f (%.5f)\ntime: %.5f" % 
                                (-results['test_n_mae'].mean(), results['test_n_mae'].std(),
                                -results['test_n_rmse'].mean(), results['test_n_rmse'].std(),
                                (end - start)))

    start = time.time()
    results = cross_validate(learner_drift, X, y, cv=kfold, scoring=scoring)
    end = time.time()
    LOGGER.info("\nModel: drift\nMAE: %.5f (%.5f), RMSE: %.5f (%.5f)\ntime: %.5f" % 
                            (-results['test_n_mae'].mean(), results['test_n_mae'].std(),
                            -results['test_n_rmse'].mean(), results['test_n_rmse'].std(),
                            (end - start)))


    # Predict
    # y_pred = cross_val_predict(learner_no_drift, X, y, cv=kfold)
"""
test_amrules_ordered_rules()