from skmultiflow.rules import AMRulesRegressor
from skmultiflow.data import AGRAWALGenerator
from skmultiflow.drift_detection import PageHinkley
from pandas_streaming.df import StreamingDataFrame
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import preprocessing
from math import sqrt
from array import array


import time

import logging
logging.basicConfig(filename='logsinfo.log', filemode='w', level=logging.INFO)
LOGGER = logging.getLogger(__name__)


# Data
#df = pd.read_csv("/home/angelo/Desktop/databases/ParisOrganized.csv")
#df = pd.read_csv("/home/angelo/Desktop/databases/used_cars_cleaned.csv")
#df = pd.read_csv("/home/angelo/Desktop/databases/cart/CartExample/cart_delve.data",
#                  names=["0","1","2","3","4","5","6","7","8","9","10"], sep="  ")
#df = pd.read_csv("/home/angelo/Desktop/databases/ailerons/Ailerons/ailerons.data",
#                        names=list(map(str, range(0,41))))
#df = pd.read_csv("/home/angelo/Desktop/databases/elevators/Elevators/elevators.data",
#                    names=list(map(str, range(0,19))))
df = pd.read_csv("/home/angelo/Desktop/databases/fried/FriedmanExample/fried_delve.data", 
                    names=list(map(str, range(0,11))), sep=" ")
"""
def test_amrules_drift():

    LOGGER.info("\ndf=\n"+str(df.head()))

    x = df.iloc[:,:-1]
    y = df.iloc[:,-1:]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    X = x_train.values.tolist()
    y = y_train.values.tolist()
    
    # scaler = preprocessing.StandardScaler().fit(X)
    # X = scaler.transform(X)
    # scaler = preprocessing.StandardScaler().fit(y)
    # y = scaler.transform(y)

    y = [item for sublist in y for item in sublist]

    # Model
    learner_no_drift = AMRulesRegressor(grace_period=200,
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=True,
                                drift_detector=None,
                                nominal_attributes=None,
                                decimal_numbers=4)

    
    learner_drift = AMRulesRegressor(grace_period=200,
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=True,
                                drift_detector=PageHinkley(),
                                nominal_attributes=None,
                                decimal_numbers=4)

    # Train
    start = time.time()
    learner_no_drift.partial_fit(X, y)
    end = time.time()
    LOGGER.info("\nno drift\nmodel=\n"+str(learner_no_drift.get_model_description())+
                "\ntime="+str(end - start))

    start = time.time()
    learner_drift.partial_fit(X, y)
    end = time.time()
    LOGGER.info("\ndrift\nmodel=\n"+str(learner_drift.get_model_description())+
                "\ntime="+str(end - start))
    
    LOGGER.info("\nsame models="+str(learner_no_drift.get_model_description() == learner_drift.get_model_description()))

    # Predict
    X = x_test.values.tolist()
    y = y_test.values.tolist()
    
    # scaler = preprocessing.StandardScaler().fit(X)
    # X = scaler.transform(X)
    # scaler = preprocessing.StandardScaler().fit(y)
    # y = scaler.transform(y)

    y = [item for sublist in y for item in sublist]
    LOGGER.info("\ny test std=\n"+str(np.std(y)))

    pred_no_drift = learner_no_drift.predict(X)
    LOGGER.info("\nno drift\nMAE="+str(round(mean_absolute_error(y, pred_no_drift),6)))
    LOGGER.info("\nno drift\nnRMSE="+str(round(sqrt(mean_squared_error(y, pred_no_drift)),6)))

    pred_drift = learner_drift.predict(X)
    LOGGER.info("\ndrift\nMAE="+str(round(mean_absolute_error(y, pred_drift),6)))
    LOGGER.info("\ndrift\nRMSE="+str(round(sqrt(mean_squared_error(y, pred_drift)),6)))

"""



def test_amrules_ordered_rules():

    #LOGGER.info("\ndf=\n"+str(df.head()))

    x = df.iloc[:,:-1]
    y = df.iloc[:,-1:]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    X = x_train.values.tolist()
    y = y_train.values.tolist()
    
    # scaler = preprocessing.StandardScaler().fit(X)
    # X = scaler.transform(X)
    # scaler = preprocessing.StandardScaler().fit(y)
    # y = scaler.transform(y)

    y = [item for sublist in y for item in sublist]
    
    # Model
    learner_no_order = AMRulesRegressor(grace_period=200,
                                 tie_threshold=0.05,
                                 expand_confidence=0.01,
                                 ordered_rules=False,
                                 drift_detector=None,
                                 nominal_attributes=None,
                                 decimal_numbers=4)

    
    learner_order = AMRulesRegressor(grace_period=200,
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=True,
                                drift_detector=None,
                                nominal_attributes=None,
                                decimal_numbers=4)

    # Train
    start = time.time()
    learner_no_order.partial_fit(X, y)
    end = time.time()
    LOGGER.info("\nno order\nmodel=\n"+str(learner_no_order.get_model_description())+
                 "\ntime="+str(end - start))

    start = time.time()
    learner_order.partial_fit(X, y)
    end = time.time()
    LOGGER.info("\norder\nmodel=\n"+str(learner_order.get_model_description())+
                "\ntime="+str(end - start))

    LOGGER.info("\nsame models="+str(learner_no_order.get_model_description() == learner_order.get_model_description()))

    # Predict
    X = x_test.values.tolist()
    y = y_test.values.tolist()
    
    # scaler = preprocessing.StandardScaler().fit(X)
    # X = scaler.transform(X)
    # scaler = preprocessing.StandardScaler().fit(y)
    # y = scaler.transform(y)

    pred_no_order = learner_no_order.predict(X)
    LOGGER.info("\nno order\nMAE="+str(round(mean_absolute_error(y, pred_no_order),6)))
    LOGGER.info("\nno order\nRMSE="+str(round(sqrt(mean_squared_error(y, pred_no_order)),6)))

    pred_order = learner_order.predict(X)
    LOGGER.info("\norder\nMAE="+str(round(mean_absolute_error(y, pred_order),6)))
    LOGGER.info("\norder\nRMSE="+str(round(sqrt(mean_squared_error(y, pred_order)),6)))




"""
def test_amrules_nominal_attributes():

    LOGGER.info("\ndf=\n"+str(df.head()))

    x = df.iloc[:,:-1]
    y = df.iloc[:,-1:]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    X = x_train.values.tolist()
    y = y_train.values.tolist()
    y = [item for sublist in y for item in sublist]
    
    # Model
    learner_no_nominal = AMRulesRegressor(grace_period=200,
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=True,
                                drift_detector=None,
                                nominal_attributes=None,
                                decimal_numbers=4)

    
    learner_nominal = AMRulesRegressor(grace_period=200,
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=True,
                                drift_detector=None,
                                nominal_attributes=[0,1,2,3,4,5,6,7,8,9],
                                decimal_numbers=4)

    # Train
    learner_no_nominal.partial_fit(X, y)
    LOGGER.info("\nno nominal\nmodel=\n"+str(learner_no_nominal.get_model_description()))

    learner_nominal.partial_fit(X, y)
    LOGGER.info("\nnominal\nmodel=\n"+str(learner_nominal.get_model_description()))

    # Predict
    X = x_test.values.tolist()
    y = y_test.values.tolist()
    y = [item for sublist in y for item in sublist]

  
    pred_no_nominal = learner_no_nominal.predict(X)
    LOGGER.info("\nno nominal\nMAE="+str(round(mean_absolute_error(y, pred_no_nominal),6)))
    LOGGER.info("\nno nominal\nnRMSE="+str(round(sqrt(mean_squared_error(y, pred_no_nominal)),6)))

    pred_nominal = learner_nominal.predict(X)
    LOGGER.info("\nnominal\nMAE="+str(round(mean_absolute_error(y, pred_nominal),6)))
    LOGGER.info("\nnominal\nRMSE="+str(round(sqrt(mean_squared_error(y, pred_nominal)),6)))

"""

"""
def test_amrules_weight_df():
    
    # Data
    path =  "/home/angelo/Desktop/databases/500_Person_Gender_Height_Weight_Index.csv"
    df = pd.read_csv(path)
    df = pd.get_dummies(df)
    LOGGER.info("\n"+str(df))

    x = df.drop(columns=["Weight"])#, "Gender"])
    y =  df["Weight"]

    LOGGER.info("\n"+str(x))
    LOGGER.info("\n"+str(y))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    X = x_train.values.tolist()
    y = y_train.values.tolist()

    # Model
    learner = AMRulesRegressor(grace_period=20, drift_detector=None, decimal_numbers=0, nominal_attributes=[1,2,3]) #

    # Train
    learner.partial_fit(X, y)
    LOGGER.info("\n"+str(learner.get_model_description()))

    # Predict
    X = x_test.values.tolist()
    y = y_test.values.tolist()

    predictions = learner.predict(X)
    
    LOGGER.info("\n"+str(y))
    LOGGER.info("\n"+str(predictions))
    p = predictions.tolist()
    LOGGER.info("\n"+str([p.count(n) for n in set(y)]))

    assert np.alltrue(predictions == y)

"""
