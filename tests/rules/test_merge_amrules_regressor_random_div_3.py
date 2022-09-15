from skmultiflow.rules import AMRulesRegressor
from skmultiflow.drift_detection import PageHinkley
import numpy as np
import pandas as pd
import sklearn.datasets as dt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

import matplotlib.pyplot as plt

import copy
import time

import logging
LOGGER = logging.getLogger(__name__)

def test_merge_amrules():

    ### random split

    #####################
    # Variables
    _PERCEPTRON = 'perceptron'
    ORDERED_RULES = True
    NUMBER_MODELS = [2,4,6,8,10,12,14,16,32]
    ARTIFICIAL_DF = False
    #_WEIGHTEDMAX = 'weighted_max'

    LOGGER.info("\n\nORDERED_RULES="+str(ORDERED_RULES)+
                "\nARTIFICIAL_DF="+str(ARTIFICIAL_DF)+"\n")

    #####################
    # Data
    if not ARTIFICIAL_DF:  
        #df = pd.read_csv("/home/angelo/Desktop/databases/cart/CartExample/cart_delve.data",
        #                  names=list(map(str, range(0,11))), sep="  ")
        #df = pd.read_csv("/home/angelo/Desktop/databases/ailerons/Ailerons/ailerons.data",
        #                names=list(map(str, range(0,41))))
        #df = pd.read_csv("/home/angelo/Desktop/databases/elevators/Elevators/elevators.data",
        #                    names=list(map(str, range(0,19))))
        #df = pd.read_csv("/home/angelo/Desktop/databases/fried/FriedmanExample/fried_delve.data", 
        #                    names=list(map(str, range(0,11))), sep=" ")
        #df = pd.read_csv("/home/angelo/Desktop/databases/trafficdeathsfinal.csv")
        #df = pd.read_csv("/home/angelo/Desktop/databases/trafficdeathsfinalbinned.csv") #to use with Historical Population
        #df = pd.read_csv("/home/angelo/Desktop/databases/diamondsfinal.csv")
        #df = pd.read_csv("/home/angelo/Desktop/databases/diamondsfinalbinned.csv") #to use with carat and x
        df = pd.read_csv("/home/angelo/Desktop/databases/usedcarsfinal.csv")      

        LOGGER.info("\ndf=\n"+str(df.head()))

        shuffled = df.sample(frac=1, axis=0, random_state=1) # random split of data
        x = shuffled.iloc[:,:-1]
        y = shuffled.iloc[:,-1:]
    else:
        x, y = dt.make_regression(n_samples=100000, 
                                    n_features=10,
                                    noise=0,
                                    shuffle=True,
                                    random_state=11) 
        LOGGER.info("\nn_samples="+str(100000)+
                    "\nn_features="+str(10)+
                    "\nnoise="+str(0)+
                    "\nshuffle="+str(True)+
                    "\nrandom_state="+str(11))

    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)


    #list_training_error = []
    list_test_mae = [] # [[2n_baseline, 2n_no_rep, 2n_best_df, 2n_best_df_no_rep], [4n_baseline, ...],[...],[...],[...]]
    list_test_rmse = []
    list_n_rules = []

    for n_models in NUMBER_MODELS:
        if not ARTIFICIAL_DF:
            X = x_train.values.tolist()
            y = y_train.values.tolist()
            y = [item for sublist in y for item in sublist]
        else:
            X = x_train
            y = y_train

        #####################
        # Model
        base_line = AMRulesRegressor(grace_period=200,
                                    tie_threshold=0.05,
                                    expand_confidence=0.01,
                                    ordered_rules=ORDERED_RULES,
                                    drift_detector=None,
                                    nominal_attributes=None,
                                    decimal_numbers=4)


        LOGGER.info("\n\ntraining "+str(n_models)+" models")

        learners = list()
        for i in range(n_models):
            learners += [
                AMRulesRegressor(grace_period=200,
                                    tie_threshold=0.05,
                                    expand_confidence=0.01,
                                    ordered_rules=ORDERED_RULES,
                                    drift_detector=None,
                                    nominal_attributes=None,
                                    decimal_numbers=4)
            ]
       
        #####################
        # Train
        start = time.time()
        base_line.partial_fit(X, y)
        end = time.time()
        #LOGGER.info("\nbaseline model=\n"+str(base_line.get_model_description())+
        #            "\ntime="+str(end - start))
        for i in range(n_models):
            l = int(len(X)/n_models)
            X_i = X[ i*l : (i+1)*l]
            y_i = y[ i*l : (i+1)*l]

            learner = learners[i]

            start = time.time()
            learner.partial_fit(X_i, y_i)
            end = time.time()
            

      
        #####################
        # Merge models

        # Baseline
        baseline_merged_learner = AMRulesRegressor(grace_period=200,
                                    tie_threshold=0.05,
                                    expand_confidence=0.01,
                                    ordered_rules=ORDERED_RULES,
                                    drift_detector=None,
                                    nominal_attributes=None,
                                    decimal_numbers=4)
        
        
        baseline_merged_learner.samples_seen += copy.deepcopy(learners[0].samples_seen)
        baseline_merged_learner.sum_target += copy.deepcopy(learners[0].sum_target)
        baseline_merged_learner.sum_target_square += copy.deepcopy(learners[0].sum_target_square)
        baseline_merged_learner.sum_attribute = copy.deepcopy(learners[0].sum_attribute)
        baseline_merged_learner.sum_attribute_squares = copy.deepcopy(learners[0].sum_attribute_squares)
        baseline_merged_learner.rule_set += copy.deepcopy(learners[0].rule_set)
        baseline_merged_learner.default_rule = copy.deepcopy(learners[0].default_rule)
        baseline_merged_learner.n_attributes_df = copy.deepcopy(learners[0].n_attributes_df)
        
        for i in range(1,n_models):
            learner = copy.deepcopy(learners[i])
            for rule in learner.rule_set:
                baseline_merged_learner.rule_set += [copy.deepcopy(rule)]
                baseline_merged_learner.samples_seen += learner.samples_seen
                baseline_merged_learner.sum_target += learner.sum_target
                baseline_merged_learner.sum_target_square += learner.sum_target_square
                baseline_merged_learner.sum_attribute = [x + y for x, y in zip(baseline_merged_learner.sum_attribute, learner.sum_attribute)] 
                baseline_merged_learner.sum_attribute_squares = [x + y for x, y in zip(baseline_merged_learner.sum_attribute_squares, \
                                                                            learner.sum_attribute_squares)] 


        
        #LOGGER.info("\n\nbaseline_merged_learner model=\n"+str(baseline_merged_learner.get_model_description()))

        
        # No rule repetition
        no_rule_rep_merged_learner = AMRulesRegressor(grace_period=200,
                                    tie_threshold=0.05,
                                    expand_confidence=0.01,
                                    ordered_rules=ORDERED_RULES,
                                    drift_detector=None,
                                    nominal_attributes=None,
                                    decimal_numbers=4)
        
        
        no_rule_rep_merged_learner.samples_seen += learners[0].samples_seen 
        no_rule_rep_merged_learner.sum_target += learners[0].sum_target
        no_rule_rep_merged_learner.sum_target_square += learners[0].sum_target_square 
        no_rule_rep_merged_learner.sum_attribute = learners[0].sum_attribute
        no_rule_rep_merged_learner.sum_attribute_squares = learners[0].sum_attribute_squares
        no_rule_rep_merged_learner.rule_set += learners[0].rule_set 
        no_rule_rep_merged_learner.default_rule = copy.deepcopy(learners[0].default_rule)
        no_rule_rep_merged_learner.n_attributes_df = copy.deepcopy(learners[0].n_attributes_df)

        for i in range(1,n_models):
            learner = copy.deepcopy(learners[i])
            for rule in learner.rule_set:
                if rule not in no_rule_rep_merged_learner.rule_set:
                    no_rule_rep_merged_learner.rule_set += [copy.deepcopy(rule)]
                    no_rule_rep_merged_learner.samples_seen += learner.samples_seen 
                    no_rule_rep_merged_learner.sum_target += learner.sum_target
                    no_rule_rep_merged_learner.sum_target_square += learner.sum_target_square 
                    no_rule_rep_merged_learner.sum_attribute = [x + y for x, y in zip(no_rule_rep_merged_learner.sum_attribute, learner.sum_attribute)] 
                    no_rule_rep_merged_learner.sum_attribute_squares = [x + y for x, y in zip(no_rule_rep_merged_learner.sum_attribute_squares, \
                                                                                learner.sum_attribute_squares)] 

                

        #LOGGER.info("\n\nno_rule_rep_merged_learner model=\n"+str(no_rule_rep_merged_learner.get_model_description()))


        # Best default rule
        best_df_rule_merged_learner = AMRulesRegressor(grace_period=200,
                                    tie_threshold=0.05,
                                    expand_confidence=0.01,
                                    ordered_rules=ORDERED_RULES,
                                    drift_detector=None,
                                    nominal_attributes=None,
                                    decimal_numbers=4)
        

        best_df_rule_merged_learner.samples_seen += learners[0].samples_seen 
        best_df_rule_merged_learner.sum_target += learners[0].sum_target
        best_df_rule_merged_learner.sum_target_square += learners[0].sum_target_square 
        best_df_rule_merged_learner.sum_attribute = learners[0].sum_attribute
        best_df_rule_merged_learner.sum_attribute_squares = learners[0].sum_attribute_squares
        best_df_rule_merged_learner.rule_set += learners[0].rule_set 
        best_df_rule_merged_learner.default_rule = copy.deepcopy(learners[0].default_rule)
        best_df_rule_merged_learner.n_attributes_df = copy.deepcopy(learners[0].n_attributes_df)

        for i in range(1,n_models):
            learner = copy.deepcopy(learners[i])
            for rule in learner.rule_set:
                best_df_rule_merged_learner.rule_set += [copy.deepcopy(rule)]
                best_df_rule_merged_learner.samples_seen += learner.samples_seen 
                best_df_rule_merged_learner.sum_target += learner.sum_target
                best_df_rule_merged_learner.sum_target_square += learner.sum_target_square 
                best_df_rule_merged_learner.sum_attribute = [x + y for x, y in zip(best_df_rule_merged_learner.sum_attribute, learner.sum_attribute)] 
                best_df_rule_merged_learner.sum_attribute_squares = [x + y for x, y in zip(best_df_rule_merged_learner.sum_attribute_squares, \
                                                                            learner.sum_attribute_squares)] 
            merged_weight = best_df_rule_merged_learner.default_rule.observed_target_stats[0]
            weight = learner.default_rule.observed_target_stats[0]
            if merged_weight < weight:
                best_df_rule_merged_learner.default_rule = copy.deepcopy(learner.default_rule)


        #LOGGER.info("\n\nnbest_df_rule_merged_learner model=\n"+str(best_df_rule_merged_learner.get_model_description()))
        

        #Best default rule and no rule repetition
        best_df_no_rep_merged_learner = AMRulesRegressor(grace_period=200,
                                    tie_threshold=0.05,
                                    expand_confidence=0.01,
                                    ordered_rules=ORDERED_RULES,
                                    drift_detector=None,
                                    nominal_attributes=None,
                                    decimal_numbers=4)
        

        best_df_no_rep_merged_learner.samples_seen += learners[0].samples_seen 
        best_df_no_rep_merged_learner.sum_target += learners[0].sum_target
        best_df_no_rep_merged_learner.sum_target_square += learners[0].sum_target_square 
        best_df_no_rep_merged_learner.sum_attribute = learners[0].sum_attribute
        best_df_no_rep_merged_learner.sum_attribute_squares = learners[0].sum_attribute_squares
        best_df_no_rep_merged_learner.rule_set += learners[0].rule_set 
        best_df_no_rep_merged_learner.default_rule = copy.deepcopy(learners[0].default_rule)
        best_df_no_rep_merged_learner.n_attributes_df = copy.deepcopy(learners[0].n_attributes_df)

        for i in range(1,n_models):
            learner = copy.deepcopy(learners[i])
            for rule in learner.rule_set:
                if rule not in best_df_no_rep_merged_learner.rule_set:
                    best_df_no_rep_merged_learner.rule_set += [copy.deepcopy(rule)]
                    best_df_no_rep_merged_learner.samples_seen += learner.samples_seen 
                    best_df_no_rep_merged_learner.sum_target += learner.sum_target
                    best_df_no_rep_merged_learner.sum_target_square += learner.sum_target_square 
                    best_df_no_rep_merged_learner.sum_attribute = [x + y for x, y in zip(best_df_no_rep_merged_learner.sum_attribute, learner.sum_attribute)] 
                    best_df_no_rep_merged_learner.sum_attribute_squares = [x + y for x, y in zip(best_df_no_rep_merged_learner.sum_attribute_squares, \
                                                                            learner.sum_attribute_squares)] 
            merged_weight = best_df_no_rep_merged_learner.default_rule.observed_target_stats[0]
            weight = learner.default_rule.observed_target_stats[0]
            if merged_weight < weight:
                best_df_no_rep_merged_learner.default_rule = copy.deepcopy(learner.default_rule)


        #LOGGER.info("\n\nbest_df_no_rep_merged_learner model=\n"+str(best_df_no_rep_merged_learner.get_model_description()))
        

        #####################
        # Predict
        mae_list = []
        rmse_list = []
        n_rules_list = []

        if not ARTIFICIAL_DF:
            X = x_test.values.tolist()
            y = y_test.values.tolist()
        else:
            X = x_test
            y = y_test

        base_line_pred = base_line.predict(X) #blue
        mae = round(mean_absolute_error(y, base_line_pred),6)
        rmse = round(sqrt(mean_squared_error(y, base_line_pred)),6)
        LOGGER.info("\nbaseline\nMAE="+str(mae)+
                    "; RMSE="+str(rmse)+"; NRules="+str(len(base_line.rule_set)))
        mae_list.append(mae)
        rmse_list.append(rmse)
        n_rules_list.append(len(base_line.rule_set))

        for i in range(n_models):
             prediction = copy.deepcopy(learners[i]).predict(X)
             #LOGGER.info("\nlearner"+str(i)+"\nMAE="+str(round(mean_absolute_error(y, prediction),6))+
             #        "; RMSE="+str(round(sqrt(mean_squared_error(y, prediction)),6))+"; NRules="+str(len(learners[i].rule_set)))
        
        
        
        # Baseline, orange
        baseline_merged_learner_pred = baseline_merged_learner.predict(X)
        mae = round(mean_absolute_error(y, baseline_merged_learner_pred),6)
        rmse = round(sqrt(mean_squared_error(y, baseline_merged_learner_pred)),6)
        LOGGER.info("\nbaseline_merged_learner\nMAE="+str(mae)+
                    "; RMSE="+str(rmse) +"; NRules="+str(len(baseline_merged_learner.rule_set)))
        mae_list.append(mae)
        rmse_list.append(rmse)
        n_rules_list.append(len(baseline_merged_learner.rule_set))
        
        
        # No rule repetition, green
        no_rule_rep_merged_learner_pred = no_rule_rep_merged_learner.predict(X)
        mae = round(mean_absolute_error(y, no_rule_rep_merged_learner_pred),6)
        rmse = round(sqrt(mean_squared_error(y, no_rule_rep_merged_learner_pred)),6)
        LOGGER.info("\nno_rule_rep_merged_learner\nMAE="+str(mae)+
                    "; RMSE="+str(rmse)+"; NRules="+str(len(no_rule_rep_merged_learner.rule_set)))
        mae_list.append(mae)
        rmse_list.append(rmse)
        n_rules_list.append(len(no_rule_rep_merged_learner.rule_set))

        
        # Best default rule, purple
        best_df_rule_merged_learner_pred = best_df_rule_merged_learner.predict(X)
        mae = round(mean_absolute_error(y, best_df_rule_merged_learner_pred),6)
        rmse = round(sqrt(mean_squared_error(y, best_df_rule_merged_learner_pred)),6)
        LOGGER.info("\nbest_df_rule_merged_learner_pred\nMAE="+str(mae)+
                    "; RMSE="+str(rmse)+"; NRules="+str(len(best_df_rule_merged_learner.rule_set)))
        mae_list.append(mae)
        rmse_list.append(rmse)
        n_rules_list.append(len(best_df_rule_merged_learner.rule_set))
        
        # Best default rule and no rule repetition, red
        best_df_no_rep_merged_learner_pred = best_df_no_rep_merged_learner.predict(X)
        mae = round(mean_absolute_error(y, best_df_no_rep_merged_learner_pred),6)
        rmse = round(sqrt(mean_squared_error(y, best_df_no_rep_merged_learner_pred)),6)
        LOGGER.info("\nbest_df_rule_and_no_rule_rep_merged_learner_pred\nMAE="+str(mae)+
                    "; RMSE="+str(rmse)+"; NRules="+str(len(best_df_no_rep_merged_learner.rule_set)))
        mae_list.append(mae)
        rmse_list.append(rmse)
        n_rules_list.append(len(best_df_no_rep_merged_learner.rule_set))

        list_test_mae.append(mae_list)
        list_test_rmse.append(rmse_list)
        list_n_rules.append(n_rules_list)
        
       
    
    plt.subplot(1,3,1)
    k=len(NUMBER_MODELS)
    n = 5
    colors_list = np.array(["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"]*k)
    n_models_list = np.array([2]*n + [4]*n + [6]*n + [8]*n + [10]*n + [12]*n + [14]*n + [16]*n + [32]*n)

    mae_l = np.array(list_test_mae).ravel()
    max_lim = int(max(mae_l) + max(mae_l)/3)
    plt.ylim([0,max_lim])
    plt.scatter(n_models_list, mae_l, c=colors_list)
    plt.xlabel('Number of nodes')
    plt.ylabel('MAE')
    plt.title('Testing MAE across nodes')
    plt.tight_layout()

    plt.subplot(1,3,2)
    rmse_l = np.array(list_test_rmse).ravel()
    max_lim = int(max(rmse_l) + max(rmse_l)/3)
    plt.ylim([0,max_lim])
    plt.scatter(n_models_list, rmse_l, c=colors_list)
    plt.xlabel('Number of nodes')
    plt.ylabel('RMSE')
    plt.title('Testing RMSE across nodes')
    plt.tight_layout()

    plt.subplot(1,3,3)
    rules_l = np.array(list_n_rules).ravel()
    max_lim = int(max(rules_l) + max(rules_l)/3)
    plt.ylim([0,max_lim])
    plt.scatter(n_models_list, rules_l, c=colors_list)
    plt.xlabel('Number of nodes')
    plt.ylabel('Number of rules')
    plt.title('Number of rules across nodes')
    plt.tight_layout()
    plt.show()
    
test_merge_amrules()
