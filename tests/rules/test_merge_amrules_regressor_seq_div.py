from skmultiflow.rules import AMRulesRegressor
from skmultiflow.data import AGRAWALGenerator
from skmultiflow.drift_detection import PageHinkley
from pandas_streaming.df import StreamingDataFrame
import numpy as np
import pandas as pd
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
# df = pd.read_csv("/Users/mariajoaolavoura/Desktop/datasets/CartExample/cart_delve.data",
#                  names=list(map(str, range(0,11))), sep="  ")
df = pd.read_csv("/Users/mariajoaolavoura/Desktop/datasets/Ailerons/ailerons.data",
                names=list(map(str, range(0,41))))
# df = pd.read_csv("/Users/mariajoaolavoura/Desktop/datasets/Elevators/elevators.data",
#                    names=list(map(str, range(0,19))))
# df = pd.read_csv("/Users/mariajoaolavoura/Desktop/datasets/FriedmanExample/fried_delve.data", 
#                    names=list(map(str, range(0,11))), sep=" ")

def test_no_rule_repetition_merge_amrules_rules():

    LOGGER.info("\ndf=\n"+str(df.head()))
    
    # Model
    base_line = AMRulesRegressor(grace_period=200,
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=True,
                                drift_detector=None,
                                nominal_attributes=None,
                                decimal_numbers=4)

    learners = list()
    n_models = 2
    LOGGER.info("\ntraining "+str(n_models)+" models")
    for i in range(n_models):
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
    
    x = df.iloc[:,:-1]
    y = df.iloc[:,-1:]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    X = x_train.values.tolist()
    y = y_train.values.tolist()
    y = [item for sublist in y for item in sublist]

    start = time.time()
    base_line.partial_fit(X, y)
    end = time.time()
    LOGGER.info("\nbaseline model=\n"+str(base_line.get_model_description())+
                "\ntime="+str(end - start))

    for i in range(n_models):
        l = int(len(X)/n_models)
        X_i = X[ i*l : (i+1)*l]
        y_i = y[ i*l : (i+1)*l]
    
        learner = learners[i]
    
        start = time.time()
        learner.partial_fit(X_i, y_i)
        end = time.time()
        # LOGGER.info("\nlearner"+str(i)+"\n"+str(learner.get_model_description())+
        #             "\ntime="+str(end - start))

    
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
    merged_learner.n_attributes_df = copy.deepcopy(learners[0].n_attributes_df)

    for i in range(1,n_models):
    
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
    X = x_test.values.tolist()
    y = y_test.values.tolist()

    base_line_pred = base_line.predict(X)
    LOGGER.info("\nbase_line\nMAE="+str(round(mean_absolute_error(y, base_line_pred),6))+
                "; RMSE="+str(round(sqrt(mean_squared_error(y, base_line_pred)),6)))

    # for i in range(n_models):
    #     prediction = learners[i].predict(X)
    #     LOGGER.info("\nlearner"+str(i)+"\nMAE="+str(round(mean_absolute_error(y, prediction),6))+
    #             "; RMSE="+str(round(sqrt(mean_squared_error(y, prediction)),6)))
    
    merged_learner_pred = merged_learner.predict(X)
    LOGGER.info("\nmerged_learner\nMAE="+str(round(mean_absolute_error(y, merged_learner_pred),6))+
                "; RMSE="+str(round(sqrt(mean_squared_error(y, merged_learner_pred)),6)))



"""
def test_kfold_no_rule_repetition_merge_amrules_rules():

    LOGGER.info("\ndf=\n"+str(df.head()))
    
    # Model
    base_line = AMRulesRegressor(grace_period=200,
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=True,
                                drift_detector=None,
                                nominal_attributes=None,
                                decimal_numbers=4)

    learners = list()
    n_models = 4
    LOGGER.info("\ntraining "+str(n_models)+" models")
    for i in range(n_models):
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
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    
    scoring = { 'n_mae':'neg_mean_absolute_error', 
                'n_rmse':'neg_root_mean_squared_error' }
    

    X = df.iloc[:,:-1]
    y = df.iloc[:,-1:]
    X = X.values.tolist()
    y = y.values.tolist()
    y = [item for sublist in y for item in sublist]


    start = time.time()
    results = cross_validate(base_line, X, y, cv=kfold, scoring=scoring)
    end = time.time()
    # LOGGER.info("\n\nModel: base line\n"+str(base_line.get_model_description()))
    LOGGER.info("\nModel: base line\nMAE: %.5f (%.5f), RMSE: %.5f (%.5f)\ntime: %.5f" % 
                                (-results['test_n_mae'].mean(), results['test_n_mae'].std(),
                                -results['test_n_rmse'].mean(), results['test_n_rmse'].std(),
                                (end - start)))


    for i in range(n_models):
        l = int(len(X)/n_models)
        X_i = X[ i*l : (i+1)*l]
        y_i = y[ i*l : (i+1)*l]
    
        learner = learners[i]

        start = time.time()
        learner.partial_fit(X_i, y_i)
        end = time.time()
        LOGGER.info("\nlearner"+str(i)+"\n"+str(learner.get_model_description())+
                    "\ntime="+str(end - start))
        

    
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
    merged_learner.n_attributes_df = copy.deepcopy(learners[0].n_attributes_df)

    for i in range(1,n_models):
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



    LOGGER.info("\n\nModel: merged\n"+str(merged_learner.get_model_description()))
    

    start = time.time()
    results = cross_val_score(merged_learner, X, y, cv=kfold, scoring=scoring)
    end = time.time()
    LOGGER.info("\nMAE: %.5f (%.5f), RMSE: %.5f (%.5f)\ntime: %.5f" % 
                                (-results['test_n_mae'].mean(), results['test_n_mae'].std(),
                                -results['test_n_rmse'].mean(), results['test_n_rmse'].std(),
                                (end - start)))

    start = time.time()
    results = cross_validate(merged_learner, X, y, cv=kfold, scoring=scoring)
    end = time.time()
    LOGGER.info("\nMAE: %.5f (%.5f), RMSE: %.5f (%.5f)\ntime: %.5f" % 
                                (-results['test_n_mae'].mean(), results['test_n_mae'].std(),
                                -results['test_n_rmse'].mean(), results['test_n_rmse'].std(),
                                (end - start)))



"""

"""
def test_merge_n_amrules_rules():
    # simple merge, just create a model with all rules from all the other models
    LOGGER.info("\ndf=\n"+str(df.head()))
    
    # Model
    base_line = AMRulesRegressor(grace_period=200,
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=True,
                                drift_detector=None,
                                nominal_attributes=None,
                                decimal_numbers=4)

    learners = list()
    n_models = 32
    LOGGER.info("\ntraining "+str(n_models)+" models")
    for i in range(n_models):
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
    
    x = df.iloc[:,:-1]
    y = df.iloc[:,-1:]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    X = x_train.values.tolist()
    y = y_train.values.tolist()
    y = [item for sublist in y for item in sublist]

    start = time.time()
    base_line.partial_fit(X, y)
    end = time.time()
    LOGGER.info("\nbaseline model=\n"+str(base_line.get_model_description())+
                "\ntime="+str(end - start))

    for i in range(n_models):
        l = int(len(X)/n_models)
        X_i = X[ i*l : (i+1)*l]
        y_i = y[ i*l : (i+1)*l]
    
        learner = learners[i]
    
        start = time.time()
        learner.partial_fit(X_i, y_i)
        end = time.time()
        LOGGER.info("\nlearner"+str(i)+"\n"+str(learner.get_model_description())+
                    "\ntime="+str(end - start))

    
    # Merge models
    merged_learner = AMRulesRegressor(grace_period=200,
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=True,
                                drift_detector=None,
                                nominal_attributes=None,
                                decimal_numbers=4)
    

    for i in range(n_models):
    
        learner = learners[i]
        merged_learner.samples_seen += learner.samples_seen 
        merged_learner.sum_target += learner.sum_target
        merged_learner.sum_target_square += learner.sum_target_square 
        merged_learner.sum_attribute = [x + y for x, y in zip(merged_learner.sum_attribute, learner.sum_attribute)] \
                                        if merged_learner.sum_attribute != [] else learner.sum_attribute
        merged_learner.sum_attribute_squares = [x + y for x, y in zip(merged_learner.sum_attribute_squares, \
                                                                    learner.sum_attribute_squares)] \
                                        if merged_learner.sum_attribute_squares != [] else learner.sum_attribute_squares

        merged_learner.rule_set += learner.rule_set 

    merged_learner.default_rule = copy.deepcopy(learners[0].default_rule)
    merged_learner.n_attributes_df = copy.deepcopy(learners[0].n_attributes_df)

    LOGGER.info("\nmerged_learner model=\n"+str(merged_learner.get_model_description()))


    # Predict
    X = x_test.values.tolist()
    y = y_test.values.tolist()

    base_line_pred = base_line.predict(X)
    LOGGER.info("\nbase_line\nMAE="+str(round(mean_absolute_error(y, base_line_pred),6))+
                "; RMSE="+str(round(sqrt(mean_squared_error(y, base_line_pred)),6)))

    for i in range(n_models):
        prediction = learners[i].predict(X)
        LOGGER.info("\nlearner"+str(i)+"\nMAE="+str(round(mean_absolute_error(y, prediction),6))+
                "; RMSE="+str(round(sqrt(mean_squared_error(y, prediction)),6)))
    
    merged_learner_pred = merged_learner.predict(X)
    LOGGER.info("\nmerged_learner\nMAE="+str(round(mean_absolute_error(y, merged_learner_pred),6))+
                "; RMSE="+str(round(sqrt(mean_squared_error(y, merged_learner_pred)),6)))

"""




"""
def test_merge_2_amrules_rules():

    LOGGER.info("\ndf=\n"+str(df.head()))

    x = df.iloc[:,:-1]
    y = df.iloc[:,-1:]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    X = x_train.values.tolist()
    y = y_train.values.tolist()
    y = [item for sublist in y for item in sublist]
    
    # Model

    base_line = AMRulesRegressor(grace_period=200,
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=True,
                                drift_detector=None,
                                nominal_attributes=None,
                                decimal_numbers=4)

    learner_1 = AMRulesRegressor(grace_period=200,
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=True,
                                drift_detector=None,
                                nominal_attributes=None,
                                decimal_numbers=4)
    
    learner_2 = AMRulesRegressor(grace_period=200,
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=True,
                                drift_detector=None,
                                nominal_attributes=None,
                                decimal_numbers=4)

    # Train
    start = time.time()
    base_line.partial_fit(X, y)
    end = time.time()
    LOGGER.info("\nbaseline model=\n"+str(base_line.get_model_description())+
                "\ntime="+str(end - start))

    X_1 = X[:int(len(X)/2)]
    X_2 = X[int(len(X)/2):]
    y_1 = y[:int(len(y)/2)]
    y_2 = y[int(len(y)/2):]

    start = time.time()
    learner_1.partial_fit(X_1, y_1)
    end = time.time()
    LOGGER.info("\nlearner_1 model=\n"+str(learner_1.get_model_description())+
                "\ntime="+str(end - start))

    start = time.time()
    learner_2.partial_fit(X_2, y_2)
    end = time.time()
    LOGGER.info("\nlearner_2 model=\n"+str(learner_2.get_model_description())+
                "\ntime="+str(end - start))

    # Merge models
    merged_learner = AMRulesRegressor(grace_period=200,
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=True,
                                drift_detector=None,
                                nominal_attributes=None,
                                decimal_numbers=4)
    

    merged_learner.samples_seen = learner_1.samples_seen + learner_2.samples_seen
    merged_learner.sum_target = learner_1.sum_target + learner_2.sum_target
    merged_learner.sum_target_square = learner_1.sum_target_square + learner_2.sum_target_square
    merged_learner.sum_attribute = [x + y for x, y in zip(learner_1.sum_attribute, learner_2.sum_attribute)]
    merged_learner.sum_attribute_squares = [x + y for x, y in zip(learner_1.sum_attribute_squares, learner_2.sum_attribute_squares)]

    merged_learner.rule_set = learner_1.rule_set + learner_2.rule_set
    merged_learner.default_rule = copy.deepcopy(learner_1.default_rule)
    merged_learner.n_attributes_df = copy.deepcopy(learner_1.n_attributes_df)

    LOGGER.info("\nmerged_learner model=\n"+str(merged_learner.get_model_description()))

    # Predict
    X = x_test.values.tolist()
    y = y_test.values.tolist()

    base_line_pred = base_line.predict(X)
    learner_1_pred = learner_1.predict(X)
    learner_2_pred = learner_2.predict(X)
    merged_learner_pred = merged_learner.predict(X)

    LOGGER.info("\nbase_line\nMAE="+str(round(mean_absolute_error(y, base_line_pred),6))+
                "; RMSE="+str(round(sqrt(mean_squared_error(y, base_line_pred)),6)))
    LOGGER.info("\nlearner_1_pred\nMAE="+str(round(mean_absolute_error(y, learner_1_pred),6))+
                "; RMSE="+str(round(sqrt(mean_squared_error(y, learner_1_pred)),6)))
    LOGGER.info("\nlearner_2_pred\nMAE="+str(round(mean_absolute_error(y, learner_2_pred),6))+
                "; RMSE="+str(round(sqrt(mean_squared_error(y, learner_2_pred)),6)))
    LOGGER.info("\nmerged_learner_pred\nMAE="+str(round(mean_absolute_error(y, merged_learner_pred),6))+
                "; RMSE="+str(round(sqrt(mean_squared_error(y, merged_learner_pred)),6)))
"""

