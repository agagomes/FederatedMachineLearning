from skmultiflow.rules import AMRulesRegressor
from skmultiflow.drift_detection import PageHinkley
import numpy as np
import pandas as pd
import sklearn.datasets as dt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from statistics import mode
from random import uniform, randint, seed

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import copy
import time

import logging
LOGGER = logging.getLogger(__name__)

def test_merge_amrules():

    ### split based on attribute, probabilistically, streaming
    ### Node vs merged

    #####################
    # Variables
    ORDERED_RULES = False
    RULE_PREDICTION = 'weighted_max' # first_hit weighted_max
    NUMBER_MODELS = [2,4,8,16,32]
    ARTIFICIAL_DF = False
    NOMINAL_ATTRIBUTES = None
    # NODE_PROBABILITY = 2/3
    seed(123) #random package

    LOGGER.info("\n\nORDERED_RULES="+str(ORDERED_RULES)+
                "\nARTIFICIAL_DF="+str(ARTIFICIAL_DF)+"\n")

    #####################
    # Data
    if not ARTIFICIAL_DF:
        df = pd.read_csv("/home/angelo/Desktop/databases/used_cars_cleaned.csv")                  
        #df = pd.read_csv("/Users/mariajoaolavoura/Desktop/datasets/CartExample/cart_delve.data",
        #                  names=list(map(str, range(0,11))), sep="  ")

        LOGGER.info("\ndf=\n"+str(df.head()))
        
        # sorting by attribute 
        df = df.sort_values('region')
        NOMINAL_ATTRIBUTES = ['region','manufacturer','fuel','title_status','transmission','drive','type','state']
        #df = df.sort_values('10') # sort by target

        x = df.iloc[:,:-1]
        y = df.iloc[:,-1:]
    else:
        x, y = dt.make_regression(n_samples=100000, 
                                    n_features=10,
                                    noise=0,
                                    shuffle=False,
                                    random_state=11) 
        LOGGER.info("\nn_samples="+str(100000)+
                    "\nn_features="+str(10)+
                    "\nnoise="+str(0)+
                    "\nshuffle="+str(False)+
                    "\nrandom_state="+str(11))

    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


    #list_training_error = []
    list_test_mae = [] # [[2n_baseline, 2n_no_rep, 2n_best_df, 2n_best_df_no_rep], [4n_baseline, ...],[...],[...],[...]]
    list_test_rmse = []
    list_n_rules = []


    if not ARTIFICIAL_DF:
        X = x_train.values.tolist()
        y = y_train.values.tolist()
        y = [item for sublist in y for item in sublist]
    else:
        X = x_train
        y = y_train


    #####################
    # Base line model
    base_line = AMRulesRegressor(grace_period=200,
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=ORDERED_RULES,
                                rule_prediction=RULE_PREDICTION,
                                drift_detector=None,
                                nominal_attributes=NOMINAL_ATTRIBUTES,
                                decimal_numbers=4)

    start = time.time()
    base_line.partial_fit(X, y)
    end = time.time()
    LOGGER.info("\nbaseline model=\n"+str(base_line.get_model_description())+
                "\ntime="+str(end - start))

    if not ARTIFICIAL_DF:
        X = x_test.values.tolist()
        y = y_test.values.tolist()
    else:
        X = x_test
        y = y_test

    base_line_pred = base_line.predict(X) #blue
    baseline_mae = round(mean_absolute_error(y, base_line_pred),6)
    baseline_rmse = round(sqrt(mean_squared_error(y, base_line_pred)),6)
    baseline_rule_set_len = len(base_line.rule_set)


    #####################
    # Learners models
    for n_models in NUMBER_MODELS:
        if not ARTIFICIAL_DF:
            X = x_train.values.tolist()
            y = y_train.values.tolist()
            y = [item for sublist in y for item in sublist]
        else:
            X = x_train
            y = y_train

        #####################
        # Models
        LOGGER.info("\n\ntraining "+str(n_models)+" models")

        learners = list()
        for i in range(n_models):
            learners += [
                AMRulesRegressor(grace_period=200,
                                    tie_threshold=0.05,
                                    expand_confidence=0.01,
                                    ordered_rules=ORDERED_RULES,
                                    rule_prediction=RULE_PREDICTION,
                                    drift_detector=None,
                                    nominal_attributes=NOMINAL_ATTRIBUTES,
                                    decimal_numbers=4)
            ]

        #####################
        # Train
        
        l = int(len(X)/n_models)
        model_idx = 0 # starting learner who has biggest probability of receiving a instance
        start = time.time()
        # for i in range(len(X)):
        #     X_i = [X[i]]
        #     y_i = [y[i]]

        #     if uniform(0, 1) <= NODE_PROBABILITY:
        #         # change learner who has biggest probability of receiving the instance
        #         if i % l:
        #             model_idx += 1
        #             # there are no more learners
        #             if model_idx >= n_models:
        #                 break # stops only this for-loop to train the learners
        #         # train    
        #         learners[model_idx].partial_fit(X_i, y_i)
        #     else:
        #         # train
        #         random_model_idx = randint(0, (n_models-1))    
        #         learners[random_model_idx].partial_fit(X_i, y_i)
        
        for i in range(len(X)):
            X_i = [X[i]]
            y_i = [y[i]]

            # change learner who has biggest probability of receiving the instance
            # l = int(len(X)/n_models)
            x_idx = (model_idx+1)*l
            #LOGGER.info("model_idx = "+str(model_idx)+"; x_idx = "+str(x_idx))
            if i == x_idx:
                model_idx += 1
                # there are no more learners
                if model_idx >= n_models:
                    # LOGGER.info(str(model_idx)+"; BREAK")
                    break # stops only this for-loop to train the learners
            # train    
            learner = learners[model_idx]
            learner.partial_fit(X_i, y_i)
            #LOGGER.info("i = "+str(i)+"/"+str(len(X)))
            #LOGGER.info("")
        end = time.time()

        # LOGGER.info("AFTER TRAIN FOR LOOP")
        
        #####################
        # Merge models

        # Baseline
        baseline_merged_learner = AMRulesRegressor(grace_period=200,
                                    tie_threshold=0.05,
                                    expand_confidence=0.01,
                                    ordered_rules=ORDERED_RULES,
                                    rule_prediction=RULE_PREDICTION,
                                    drift_detector=None,
                                    nominal_attributes=NOMINAL_ATTRIBUTES,
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



        LOGGER.info("\n\nbaseline_merged_learner model=\n"+str(baseline_merged_learner.get_model_description()))


        # # No rule repetition
        # no_rule_rep_merged_learner = AMRulesRegressor(grace_period=200,
        #                             tie_threshold=0.05,
        #                             expand_confidence=0.01,
        #                             ordered_rules=ORDERED_RULES,
        #                             drift_detector=None,
        #                             nominal_attributes=NOMINAL_ATTRIBUTES,
        #                             decimal_numbers=4)
        

        # no_rule_rep_merged_learner.samples_seen += learners[0].samples_seen 
        # no_rule_rep_merged_learner.sum_target += learners[0].sum_target
        # no_rule_rep_merged_learner.sum_target_square += learners[0].sum_target_square 
        # no_rule_rep_merged_learner.sum_attribute = learners[0].sum_attribute
        # no_rule_rep_merged_learner.sum_attribute_squares = learners[0].sum_attribute_squares
        # no_rule_rep_merged_learner.rule_set += learners[0].rule_set 
        # no_rule_rep_merged_learner.default_rule = copy.deepcopy(learners[0].default_rule)
        # no_rule_rep_merged_learner.n_attributes_df = copy.deepcopy(learners[0].n_attributes_df)

        # for i in range(1,n_models):
        #     learner = copy.deepcopy(learners[i])
        #     for rule in learner.rule_set:
        #         if rule not in no_rule_rep_merged_learner.rule_set:
        #             no_rule_rep_merged_learner.rule_set += [copy.deepcopy(rule)]
        #             no_rule_rep_merged_learner.samples_seen += learner.samples_seen 
        #             no_rule_rep_merged_learner.sum_target += learner.sum_target
        #             no_rule_rep_merged_learner.sum_target_square += learner.sum_target_square 
        #             no_rule_rep_merged_learner.sum_attribute = [x + y for x, y in zip(no_rule_rep_merged_learner.sum_attribute, learner.sum_attribute)] 
        #             no_rule_rep_merged_learner.sum_attribute_squares = [x + y for x, y in zip(no_rule_rep_merged_learner.sum_attribute_squares, \
        #                                                                         learner.sum_attribute_squares)] 

                

        # LOGGER.info("\n\nno_rule_rep_merged_learner model=\n"+str(no_rule_rep_merged_learner.get_model_description()))


        # # Best default rule
        # best_df_rule_merged_learner = AMRulesRegressor(grace_period=200,
        #                             tie_threshold=0.05,
        #                             expand_confidence=0.01,
        #                             ordered_rules=ORDERED_RULES,
        #                             drift_detector=None,
        #                             nominal_attributes=NOMINAL_ATTRIBUTES,
        #                             decimal_numbers=4)
        

        # best_df_rule_merged_learner.samples_seen += learners[0].samples_seen 
        # best_df_rule_merged_learner.sum_target += learners[0].sum_target
        # best_df_rule_merged_learner.sum_target_square += learners[0].sum_target_square 
        # best_df_rule_merged_learner.sum_attribute = learners[0].sum_attribute
        # best_df_rule_merged_learner.sum_attribute_squares = learners[0].sum_attribute_squares
        # best_df_rule_merged_learner.rule_set += learners[0].rule_set 
        # best_df_rule_merged_learner.default_rule = copy.deepcopy(learners[0].default_rule)
        # best_df_rule_merged_learner.n_attributes_df = copy.deepcopy(learners[0].n_attributes_df)

        # for i in range(1,n_models):
        #     learner = copy.deepcopy(learners[i])
        #     for rule in learner.rule_set:
        #         best_df_rule_merged_learner.rule_set += [copy.deepcopy(rule)]
        #         best_df_rule_merged_learner.samples_seen += learner.samples_seen 
        #         best_df_rule_merged_learner.sum_target += learner.sum_target
        #         best_df_rule_merged_learner.sum_target_square += learner.sum_target_square 
        #         best_df_rule_merged_learner.sum_attribute = [x + y for x, y in zip(best_df_rule_merged_learner.sum_attribute, learner.sum_attribute)] 
        #         best_df_rule_merged_learner.sum_attribute_squares = [x + y for x, y in zip(best_df_rule_merged_learner.sum_attribute_squares, \
        #                                                                     learner.sum_attribute_squares)] 
        #     merged_weight = best_df_rule_merged_learner.default_rule.observed_target_stats[0]
        #     weight = learner.default_rule.observed_target_stats[0]
        #     if merged_weight < weight:
        #         best_df_rule_merged_learner.default_rule = copy.deepcopy(learner.default_rule)


        # LOGGER.info("\n\nbest_df_rule_merged_learner model=\n"+str(best_df_rule_merged_learner.get_model_description()))


        # Best default rule and no rule repetition
        best_df_no_rep_merged_learner = AMRulesRegressor(grace_period=200,
                                    tie_threshold=0.05,
                                    expand_confidence=0.01,
                                    ordered_rules=ORDERED_RULES,
                                    rule_prediction=RULE_PREDICTION,
                                    drift_detector=None,
                                    nominal_attributes=NOMINAL_ATTRIBUTES,
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


        LOGGER.info("\n\nbest_df_no_rep_merged_learner model=\n"+str(best_df_no_rep_merged_learner.get_model_description()))


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


        # Nodes
        for i in range(n_models):
            node_pred = learners[i].predict(X)
            mae = round(mean_absolute_error(y, node_pred),6)
            rmse = round(sqrt(mean_squared_error(y, node_pred)),6)
            # LOGGER.info("\nlearner"+str(i)+"\nMAE="+str(mae)+
            #             "; RMSE="+str(rmse))
            mae_list.append(mae)
            rmse_list.append(rmse)
            n_rules_list.append(len(learners[i].rule_set))
        
        LOGGER.info("\n\nn_rules_list learners\n"+str(n_rules_list))


        # list_test_mae.append(mae_list)
        # list_test_rmse.append(rmse_list)
        # list_n_rules.append(n_rules_list)


        # Centralized
        LOGGER.info("\nbaseline\nMAE="+str(baseline_mae)+
                    "; RMSE="+str(baseline_rmse))
        mae_list.append(baseline_mae)
        rmse_list.append(baseline_rmse)
        n_rules_list.append(baseline_rule_set_len)

        
        # Baseline, orange
        baseline_merged_learner_pred = baseline_merged_learner.predict(X)
        mae = round(mean_absolute_error(y, baseline_merged_learner_pred),6)
        rmse = round(sqrt(mean_squared_error(y, baseline_merged_learner_pred)),6)
        LOGGER.info("\nbaseline_merged_learner\nMAE="+str(mae)+
                    "; RMSE="+str(rmse))
        mae_list.append(mae)
        rmse_list.append(rmse)
        n_rules_list.append(len(baseline_merged_learner.rule_set))


        # # No rule repetition, green
        # no_rule_rep_merged_learner_pred = no_rule_rep_merged_learner.predict(X)
        # mae = round(mean_absolute_error(y, no_rule_rep_merged_learner_pred),6)
        # rmse = round(sqrt(mean_squared_error(y, no_rule_rep_merged_learner_pred)),6)
        # LOGGER.info("\nno_rule_rep_merged_learner\nMAE="+str(mae)+
        #             "; RMSE="+str(rmse))
        # mae_list.append(mae)
        # rmse_list.append(rmse)
        # n_rules_list.append(len(no_rule_rep_merged_learner.rule_set))

        
        # # Best default rule, purple
        # best_df_rule_merged_learner_pred = best_df_rule_merged_learner.predict(X)
        # mae = round(mean_absolute_error(y, best_df_rule_merged_learner_pred),6)
        # rmse = round(sqrt(mean_squared_error(y, best_df_rule_merged_learner_pred)),6)
        # LOGGER.info("\nbest_df_rule_merged_learner_pred\nMAE="+str(mae)+
        #             "; RMSE="+str(rmse))
        # mae_list.append(mae)
        # rmse_list.append(rmse)
        # n_rules_list.append(len(best_df_rule_merged_learner.rule_set))


        # Best default rule and no rule repetition, red
        best_df_no_rep_merged_learner_pred = best_df_no_rep_merged_learner.predict(X)
        mae = round(mean_absolute_error(y, best_df_no_rep_merged_learner_pred),6)
        rmse = round(sqrt(mean_squared_error(y, best_df_no_rep_merged_learner_pred)),6)
        LOGGER.info("\nbest_df_rule_merged_learner_pred\nMAE="+str(mae)+
                    "; RMSE="+str(rmse))
        mae_list.append(mae)
        rmse_list.append(rmse)
        n_rules_list.append(len(best_df_no_rep_merged_learner.rule_set))

        list_test_mae.append(mae_list)
        list_test_rmse.append(rmse_list)
        list_n_rules.append(n_rules_list)


    
    # n = 5
    # colors_list = np.array(["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"]*n)
    # n_models_list = np.array([2]*n + [4]*n + [8]*n + [16]*n + [32]*n)

    colors_list = []
    n_models_list = []
    for i in range(len(NUMBER_MODELS)):
        colors_list += ["tab:green"]*NUMBER_MODELS[i] + ["tab:blue", "tab:orange", "tab:red"]
        n_models_list += [NUMBER_MODELS[i]]*(NUMBER_MODELS[i]+3)
    colors_list = np.array(colors_list)
    n_models_list = np.array(n_models_list)

    plt.subplot(1,3,1)
    mae_l = sum(list_test_mae, []) # np.array(list_test_mae).ravel()
    max_lim = int(max(mae_l) + max(mae_l)/3)
    plt.ylim([0,max_lim])
    plt.scatter(n_models_list, mae_l, c=colors_list)
    plt.xlabel('Number of nodes')
    plt.ylabel('MAE')
    plt.title('Testing MAE across nodes')
    plt.tight_layout()

    plt.subplot(1,3,2)
    rmse_l = sum(list_test_rmse, []) # np.array(list_test_rmse).ravel()
    max_lim = int(max(rmse_l) + max(rmse_l)/3)
    plt.ylim([0,max_lim])
    plt.scatter(n_models_list, rmse_l, c=colors_list)
    plt.xlabel('Number of nodes')
    plt.ylabel('RMSE')
    plt.title('Testing RMSE across nodes')
    plt.tight_layout()

    plt.subplot(1,3,3)
    rules_l = sum(list_n_rules, []) # np.array(list_n_rules).ravel()
    max_lim = int(max(rules_l) + max(rules_l)/3)
    plt.ylim([0,max_lim])
    plt.scatter(n_models_list, rules_l, c=colors_list)
    plt.xlabel('Number of nodes')
    plt.ylabel('Number of rules')
    plt.title('Number of rules across nodes')
    plt.tight_layout()
    plt.show()
test_merge_amrules()