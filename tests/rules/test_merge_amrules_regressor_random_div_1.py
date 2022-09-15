from skmultiflow.rules import AMRulesRegressor
from skmultiflow.data import AGRAWALGenerator
from skmultiflow.drift_detection import PageHinkley
from pandas_streaming.df import StreamingDataFrame
import numpy as np
import pandas as pd
import sklearn.datasets as dt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_validate, cross_val_score
from sklearn import preprocessing
from math import sqrt
from array import array

import copy
import time

import logging
LOGGER = logging.getLogger(__name__)


# Data
#df = pd.read_csv("/home/angelo/Desktop/databases/cart/CartExample/cart_delve.data",
#                  names=["0","1","2","3","4","5","6","7","8","9","10"], sep="  ")
#df = pd.read_csv("/home/angelo/Desktop/databases/ailerons/Ailerons/ailerons.data",
#                        names=list(map(str, range(0,41))))
#df = pd.read_csv("/home/angelo/Desktop/databases/elevators/Elevators/elevators.data",
#                    names=list(map(str, range(0,19))))
#df = pd.read_csv("/home/angelo/Desktop/databases/fried/FriedmanExample/fried_delve.data", 
#                    names=list(map(str, range(0,11))), sep=" ")

# LOGGER.info("\ndf=\n"+str(df.head()))

# shuffled = df.sample(frac=1, axis=0, random_state=1) # random split of data

n_samples=40000 
n_features=2
noise=100
shuffle=True
random_state=11
x, y = dt.make_regression(n_samples=n_samples, 
                            n_features=n_features,
                            noise=noise,
                            shuffle=shuffle,
                            random_state=random_state) 
LOGGER.info("\nn_samples="+str(n_samples)+
            "\nn_features="+str(n_features)+
            "\nnoise="+str(noise)+
            "\nshuffle="+str(shuffle)+
            "\nrandom_state="+str(random_state))

# x = shuffled.iloc[:,:-1]
# y = shuffled.iloc[:,-1:]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

# X = x_train.values.tolist()
# y = y_train.values.tolist()
# y = [item for sublist in y for item in sublist]
# X = x_train
# y = y_train

NUMBER_MODELS = 32

def test_baseline_merge_amrules():

    # Model
    base_line = AMRulesRegressor(grace_period=200,
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=True,
                                drift_detector=None,
                                nominal_attributes=None,
                                decimal_numbers=4)

    
    LOGGER.info("\ntraining "+str(NUMBER_MODELS)+" models")

    learners = list()
    for i in range(NUMBER_MODELS):
        learners += [
            AMRulesRegressor(grace_period=200,
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=True,
                                drift_detector=None,
                                nominal_attributes=None,
                                decimal_numbers=4)
        ]


    # Train
    X = x_train
    y = y_train

    start = time.time()
    base_line.partial_fit(X, y)
    end = time.time()
    LOGGER.info("\nbaseline model=\n"+str(base_line.get_model_description())+
                "\ntime="+str(end - start))

    for i in range(NUMBER_MODELS):
        l = int(len(X)/NUMBER_MODELS)
        X_i = X[ i*l : (i+1)*l]
        y_i = y[ i*l : (i+1)*l]

        learner = learners[i]

        start = time.time()
        learner.partial_fit(X_i, y_i)
        end = time.time()

        
    # Merge models
    merged_learner = AMRulesRegressor(grace_period=200,
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=True,
                                drift_detector=None,
                                nominal_attributes=None,
                                decimal_numbers=4)
    

    merged_learner.samples_seen += learners[0].samples_seen
    merged_learner.sum_target += learners[0].sum_target
    merged_learner.sum_target_square += learners[0].sum_target_square
    merged_learner.sum_attribute = learners[0].sum_attribute
    merged_learner.sum_attribute_squares = learners[0].sum_attribute_squares
    merged_learner.rule_set += learners[0].rule_set
    merged_learner.default_rule = copy.deepcopy(learners[0].default_rule)
    merged_learner.n_attributes_df = learners[0].n_attributes_df

    for i in range(1,NUMBER_MODELS):
        learner = learners[i]
        for rule in learner.rule_set:
            merged_learner.rule_set += [copy.deepcopy(rule)]
            merged_learner.samples_seen += learner.samples_seen
            merged_learner.sum_target += learner.sum_target
            merged_learner.sum_target_square += learner.sum_target_square
            merged_learner.sum_attribute = [x + y for x, y in zip(merged_learner.sum_attribute, learner.sum_attribute)] 
            merged_learner.sum_attribute_squares = [x + y for x, y in zip(merged_learner.sum_attribute_squares, \
                                                                        learner.sum_attribute_squares)] 



    LOGGER.info("\n\nmerged_learner model=\n"+str(merged_learner.get_model_description()))


    # Predict
    # X = x_test.values.tolist()
    # y = y_test.values.tolist()
    X = x_test
    y = y_test

    base_line_pred = base_line.predict(X)
    LOGGER.info("\nbase_line\nMAE="+str(round(mean_absolute_error(y, base_line_pred),6))+
                "; RMSE="+str(round(sqrt(mean_squared_error(y, base_line_pred)),6)))

    # for i in range(NUMBER_MODELS):
    #     prediction = learners[i].predict(X)
    #     LOGGER.info("\nlearner"+str(i)+"\nMAE="+str(round(mean_absolute_error(y, prediction),6))+
    #             "; RMSE="+str(round(sqrt(mean_squared_error(y, prediction)),6)))
    
    merged_learner_pred = merged_learner.predict(X)
    LOGGER.info("\nmerged_learner\nMAE="+str(round(mean_absolute_error(y, merged_learner_pred),6))+
                "; RMSE="+str(round(sqrt(mean_squared_error(y, merged_learner_pred)),6)))




def test_no_rule_repetition_merge_amrules():

    # Model
    base_line = AMRulesRegressor(grace_period=200,
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=True,
                                drift_detector=None,
                                nominal_attributes=None,
                                decimal_numbers=4)

    LOGGER.info("\ntraining "+str(NUMBER_MODELS)+" models")

    learners = list()
    for i in range(NUMBER_MODELS):
        learners += [
            AMRulesRegressor(grace_period=200,
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=True,
                                drift_detector=None,
                                nominal_attributes=None,
                                decimal_numbers=4)
        ]


    # Train
    X = x_train
    y = y_train

    start = time.time()
    base_line.partial_fit(X, y)
    end = time.time()
    LOGGER.info("\nbaseline model=\n"+str(base_line.get_model_description())+
                "\ntime="+str(end - start))

    for i in range(NUMBER_MODELS):
        l = int(len(X)/NUMBER_MODELS)
        X_i = X[ i*l : (i+1)*l]
        y_i = y[ i*l : (i+1)*l]

        learner = learners[i]

        start = time.time()
        learner.partial_fit(X_i, y_i)
        end = time.time()


    # Merge models
    merged_learner = AMRulesRegressor(grace_period=200,
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=True,
                                drift_detector=None,
                                nominal_attributes=None,
                                decimal_numbers=4)
    

    merged_learner.samples_seen += learners[0].samples_seen 
    merged_learner.sum_target += learners[0].sum_target
    merged_learner.sum_target_square += learners[0].sum_target_square 
    merged_learner.sum_attribute = learners[0].sum_attribute
    merged_learner.sum_attribute_squares = learners[0].sum_attribute_squares
    merged_learner.rule_set += learners[0].rule_set 
    merged_learner.default_rule = copy.deepcopy(learners[0].default_rule)
    merged_learner.n_attributes_df = learners[0].n_attributes_df

    for i in range(1,NUMBER_MODELS):
        learner = learners[i]
        for rule in learner.rule_set:
            if rule not in merged_learner.rule_set:
                merged_learner.rule_set += [copy.deepcopy(rule)]
                merged_learner.samples_seen += learner.samples_seen 
                merged_learner.sum_target += learner.sum_target
                merged_learner.sum_target_square += learner.sum_target_square 
                merged_learner.sum_attribute = [x + y for x, y in zip(merged_learner.sum_attribute, learner.sum_attribute)] 
                merged_learner.sum_attribute_squares = [x + y for x, y in zip(merged_learner.sum_attribute_squares, \
                                                                            learner.sum_attribute_squares)] 

            

    LOGGER.info("\n\nmerged_learner model=\n"+str(merged_learner.get_model_description()))


    # Predict
    # X = x_test.values.tolist()
    # y = y_test.values.tolist()
    X = x_test
    y = y_test

    base_line_pred = base_line.predict(X)
    LOGGER.info("\nbase_line\nMAE="+str(round(mean_absolute_error(y, base_line_pred),6))+
                "; RMSE="+str(round(sqrt(mean_squared_error(y, base_line_pred)),6)))

    # for i in range(NUMBER_MODELS):
    #     prediction = learners[i].predict(X)
    #     LOGGER.info("\nlearner"+str(i)+"\nMAE="+str(round(mean_absolute_error(y, prediction),6))+
    #             "; RMSE="+str(round(sqrt(mean_squared_error(y, prediction)),6)))
    
    merged_learner_pred = merged_learner.predict(X)
    LOGGER.info("\nmerged_learner\nMAE="+str(round(mean_absolute_error(y, merged_learner_pred),6))+
                "; RMSE="+str(round(sqrt(mean_squared_error(y, merged_learner_pred)),6)))




def test_best_default_rule_merge_amrules():
    # Model
    base_line = AMRulesRegressor(grace_period=200,
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=True,
                                drift_detector=None,
                                nominal_attributes=None,
                                decimal_numbers=4)

    LOGGER.info("\ntraining "+str(NUMBER_MODELS)+" models")

    learners = list()
    for i in range(NUMBER_MODELS):
        learners += [
            AMRulesRegressor(grace_period=200,
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=True,
                                drift_detector=None,
                                nominal_attributes=None,
                                decimal_numbers=4)
        ]


    # Train
    X = x_train
    y = y_train
    
    start = time.time()
    base_line.partial_fit(X, y)
    end = time.time()
    LOGGER.info("\nbaseline model=\n"+str(base_line.get_model_description())+
                "\ntime="+str(end - start))

    for i in range(NUMBER_MODELS):
        l = int(len(X)/NUMBER_MODELS)
        X_i = X[ i*l : (i+1)*l]
        y_i = y[ i*l : (i+1)*l]

        learner = learners[i]

        start = time.time()
        learner.partial_fit(X_i, y_i)
        end = time.time()

    # Merge models
    merged_learner = AMRulesRegressor(grace_period=200,
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=True,
                                drift_detector=None,
                                nominal_attributes=None,
                                decimal_numbers=4)
    

    merged_learner.samples_seen += learners[0].samples_seen 
    merged_learner.sum_target += learners[0].sum_target
    merged_learner.sum_target_square += learners[0].sum_target_square 
    merged_learner.sum_attribute = learners[0].sum_attribute
    merged_learner.sum_attribute_squares = learners[0].sum_attribute_squares
    merged_learner.rule_set += learners[0].rule_set 
    merged_learner.default_rule = copy.deepcopy(learners[0].default_rule)
    merged_learner.n_attributes_df = learners[0].n_attributes_df

    for i in range(1,NUMBER_MODELS):
        learner = learners[i]
        for rule in learner.rule_set:
            merged_learner.rule_set += [copy.deepcopy(rule)]
            merged_learner.samples_seen += learner.samples_seen 
            merged_learner.sum_target += learner.sum_target
            merged_learner.sum_target_square += learner.sum_target_square 
            merged_learner.sum_attribute = [x + y for x, y in zip(merged_learner.sum_attribute, learner.sum_attribute)] 
            merged_learner.sum_attribute_squares = [x + y for x, y in zip(merged_learner.sum_attribute_squares, \
                                                                        learner.sum_attribute_squares)] 
        merged_weight = merged_learner.default_rule.observed_target_stats[0]
        weight = learner.default_rule.observed_target_stats[0]
        if merged_weight < weight:
            merged_learner.default_rule = copy.deepcopy(learner.default_rule)


    LOGGER.info("\n\nmerged_learner model=\n"+str(merged_learner.get_model_description()))


    # Predict
    # X = x_test.values.tolist()
    # y = y_test.values.tolist()
    X = x_test
    y = y_test

    base_line_pred = base_line.predict(X)
    LOGGER.info("\nbase_line\nMAE="+str(round(mean_absolute_error(y, base_line_pred),6))+
                "; RMSE="+str(round(sqrt(mean_squared_error(y, base_line_pred)),6)))

    # for i in range(NUMBER_MODELS):
    #     prediction = learners[i].predict(X)
    #     LOGGER.info("\nlearner"+str(i)+"\nMAE="+str(round(mean_absolute_error(y, prediction),6))+
    #             "; RMSE="+str(round(sqrt(mean_squared_error(y, prediction)),6)))
    
    merged_learner_pred = merged_learner.predict(X)
    LOGGER.info("\nmerged_learner\nMAE="+str(round(mean_absolute_error(y, merged_learner_pred),6))+
                "; RMSE="+str(round(sqrt(mean_squared_error(y, merged_learner_pred)),6)))




def test_best_default_no_rule_repetition_rule_merge_amrules():

    # Model
    base_line = AMRulesRegressor(grace_period=200,
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=True,
                                drift_detector=None,
                                nominal_attributes=None,
                                decimal_numbers=4)

    LOGGER.info("\ntraining "+str(NUMBER_MODELS)+" models")

    learners = list()
    for i in range(NUMBER_MODELS):
        learners += [
            AMRulesRegressor(grace_period=200,
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=True,
                                drift_detector=None,
                                nominal_attributes=None,
                                decimal_numbers=4)
        ]


    # Train
    X = x_train
    y = y_train
    
    start = time.time()
    base_line.partial_fit(X, y)
    end = time.time()
    LOGGER.info("\nbaseline model=\n"+str(base_line.get_model_description())+
                "\ntime="+str(end - start))

    for i in range(NUMBER_MODELS):
        l = int(len(X)/NUMBER_MODELS)
        X_i = X[ i*l : (i+1)*l]
        y_i = y[ i*l : (i+1)*l]

        learner = learners[i]

        start = time.time()
        learner.partial_fit(X_i, y_i)
        end = time.time()


    # Merge models
    merged_learner = AMRulesRegressor(grace_period=200,
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=True,
                                drift_detector=None,
                                nominal_attributes=None,
                                decimal_numbers=4)
    

    merged_learner.samples_seen += learners[0].samples_seen 
    merged_learner.sum_target += learners[0].sum_target
    merged_learner.sum_target_square += learners[0].sum_target_square 
    merged_learner.sum_attribute = learners[0].sum_attribute
    merged_learner.sum_attribute_squares = learners[0].sum_attribute_squares
    merged_learner.rule_set += learners[0].rule_set 
    merged_learner.default_rule = copy.deepcopy(learners[0].default_rule)
    merged_learner.n_attributes_df = learners[0].n_attributes_df

    for i in range(1,NUMBER_MODELS):
        learner = learners[i]
        for rule in learner.rule_set:
            if rule not in merged_learner.rule_set:
                merged_learner.rule_set += [copy.deepcopy(rule)]
                merged_learner.samples_seen += learner.samples_seen 
                merged_learner.sum_target += learner.sum_target
                merged_learner.sum_target_square += learner.sum_target_square 
                merged_learner.sum_attribute = [x + y for x, y in zip(merged_learner.sum_attribute, learner.sum_attribute)] 
                merged_learner.sum_attribute_squares = [x + y for x, y in zip(merged_learner.sum_attribute_squares, \
                                                                        learner.sum_attribute_squares)] 
        merged_weight = merged_learner.default_rule.observed_target_stats[0]
        weight = learner.default_rule.observed_target_stats[0]
        if merged_weight < weight:
            merged_learner.default_rule = copy.deepcopy(learner.default_rule)


    LOGGER.info("\n\nmerged_learner model=\n"+str(merged_learner.get_model_description()))


    # Predict
    # X = x_test.values.tolist()
    # y = y_test.values.tolist()
    X = x_test
    y = y_test

    base_line_pred = base_line.predict(X)
    LOGGER.info("\nbase_line\nMAE="+str(round(mean_absolute_error(y, base_line_pred),6))+
                "; RMSE="+str(round(sqrt(mean_squared_error(y, base_line_pred)),6)))

    # for i in range(NUMBER_MODELS):
    #     prediction = learners[i].predict(X)
    #     LOGGER.info("\nlearner"+str(i)+"\nMAE="+str(round(mean_absolute_error(y, prediction),6))+
    #             "; RMSE="+str(round(sqrt(mean_squared_error(y, prediction)),6)))
    
    merged_learner_pred = merged_learner.predict(X)
    LOGGER.info("\nmerged_learner\nMAE="+str(round(mean_absolute_error(y, merged_learner_pred),6))+
                "; RMSE="+str(round(sqrt(mean_squared_error(y, merged_learner_pred)),6)))

test_baseline_merge_amrules()

