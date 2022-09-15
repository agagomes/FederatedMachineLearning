from skmultiflow.rules import AMRulesRegressor
from skmultiflow.drift_detection import PageHinkley
import numpy as np
import pandas as pd
import sklearn.datasets as dt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from math import sqrt
from statistics import mean

import matplotlib.pyplot as plt

import copy
import time

import logging
LOGGER = logging.getLogger(__name__)

def test_merge_amrules():

    ### cross validation

    #####################
    # Variables
    ORDERED_RULES = False
    NUMBER_MODELS = [2,4,8,16,32]
    ARTIFICIAL_DF = False

    LOGGER.info("\n\nORDERED_RULES="+str(ORDERED_RULES)+
                "\nARTIFICIAL_DF="+str(ARTIFICIAL_DF)+"\n")

    #####################
    # Data
    if not ARTIFICIAL_DF:
        df = pd.read_csv("/Users/mariajoaolavoura/Desktop/datasets/CartExample/cart_delve.data",
                          names=list(map(str, range(0,11))), sep="  ")
        #df = pd.read_csv("/Users/mariajoaolavoura/Desktop/datasets/Ailerons/ailerons.data",
        #                names=list(map(str, range(0,41))))
        #df = pd.read_csv("/Users/mariajoaolavoura/Desktop/datasets/Elevators/elevators.data",
        #                    names=list(map(str, range(0,19))))
        #df = pd.read_csv("/Users/mariajoaolavoura/Desktop/datasets/FriedmanExample/fried_delve.data", 
        #                    names=list(map(str, range(0,11))), sep=" ")

        LOGGER.info("\ndf=\n"+str(df.head()))

        #shuffled = df.sample(frac=1, axis=0, random_state=1) # random split of data
        X = df.iloc[:,:-1]
        y = df.iloc[:,-1:]
    else:
        X, y = dt.make_regression(n_samples=100000, 
                                    n_features=10,
                                    noise=0,
                                    shuffle=True,
                                    random_state=11) 
        LOGGER.info("\nn_samples="+str(100000)+
                    "\nn_features="+str(10)+
                    "\nnoise="+str(0)+
                    "\nshuffle="+str(True)+
                    "\nrandom_state="+str(11))

    
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    if not ARTIFICIAL_DF:
        X = np.array(X.values.tolist())
        y = y.values.tolist()
        y = np.array([item for sublist in y for item in sublist])

    #list_training_error = []
    list_test_mae = [] # [[2n_baseline, 2n_no_rep, 2n_best_df, 2n_best_df_no_rep], [4n_baseline, ...],[...],[...],[...]]
    list_test_rmse = []
    list_n_rules = []

    for n_models in NUMBER_MODELS:
        
        mae_temp = []
        rmse_temp = []
        nrules_temp = []

        #####################
        # Model

        # Centralized baseline
        centralized = AMRulesRegressor(grace_period=200,
                                    tie_threshold=0.05,
                                    expand_confidence=0.01,
                                    ordered_rules=ORDERED_RULES,
                                    drift_detector=None,
                                    nominal_attributes=None,
                                    decimal_numbers=4)


        LOGGER.info("\n\ntraining "+str(n_models)+" models")

        # Nodes
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

        
        # Merge models - Baseline
        baseline_merged_learner = AMRulesRegressor(grace_period=200,
                                    tie_threshold=0.05,
                                    expand_confidence=0.01,
                                    ordered_rules=ORDERED_RULES,
                                    drift_detector=None,
                                    nominal_attributes=None,
                                    decimal_numbers=4)

        # Merge models - No rule repetition
        no_rule_rep_merged_learner = AMRulesRegressor(grace_period=200,
                                    tie_threshold=0.05,
                                    expand_confidence=0.01,
                                    ordered_rules=ORDERED_RULES,
                                    drift_detector=None,
                                    nominal_attributes=None,
                                    decimal_numbers=4)

        # Merge models - Best default rule
        best_df_rule_merged_learner = AMRulesRegressor(grace_period=200,
                                    tie_threshold=0.05,
                                    expand_confidence=0.01,
                                    ordered_rules=ORDERED_RULES,
                                    drift_detector=None,
                                    nominal_attributes=None,
                                    decimal_numbers=4)

        # Merge models - Best default rule and no rule repetition
        best_df_no_rep_merged_learner = AMRulesRegressor(grace_period=200,
                                    tie_threshold=0.05,
                                    expand_confidence=0.01,
                                    ordered_rules=ORDERED_RULES,
                                    drift_detector=None,
                                    nominal_attributes=None,
                                    decimal_numbers=4)

        #####################
        # Train

        kf = KFold(n_splits=3, random_state=7, shuffle=True)
        for train_index, test_index in kf.split(X):
            train_x, test_x = X[train_index], X[test_index]
            train_y, test_y = y[train_index], y[test_index]

            start = time.time()
            centralized.partial_fit(train_x, train_y)
            end = time.time()
            # LOGGER.info("\nbaseline model=\n"+str(centralized.get_model_description())+
            #            "\ntime="+str(end - start))

            for i in range(n_models):
                l = int(len(X)/n_models)
                train_xi = train_x[ i*l : (i+1)*l]
                train_yi = train_y[ i*l : (i+1)*l]

                learner = learners[i]

                start = time.time()
                learner.partial_fit(train_xi, train_yi)
                end = time.time()



            #####################
            # Merge models

            # baseline_merged_learner
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


            # no_rule_rep_merged_learner
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


            # best_df_rule_merged_learner
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


            # best_df_no_rep_merged_learner
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

            centralized_pred = centralized.predict(test_x) #blue
            mae = round(mean_absolute_error(test_y, centralized_pred),6)
            rmse = round(sqrt(mean_squared_error(test_y, centralized_pred)),6)
            LOGGER.info("\ncentralized\nMAE="+str(mae)+
                        "; RMSE="+str(rmse))
            mae_list.append(mae)
            rmse_list.append(rmse)
            n_rules_list.append(len(centralized.rule_set))

            # for i in range(n_models):
            #     prediction = copy.deepcopy(learners[i]).predict(X)
            #     LOGGER.info("\nlearner"+str(i)+"\nMAE="+str(round(mean_absolute_error(y, prediction),6))+
            #             "; RMSE="+str(round(sqrt(mean_squared_error(y, prediction)),6)))
            
            
            # Baseline, orange
            baseline_merged_learner_pred = baseline_merged_learner.predict(test_x)
            mae = round(mean_absolute_error(test_y, baseline_merged_learner_pred),6)
            rmse = round(sqrt(mean_squared_error(test_y, baseline_merged_learner_pred)),6)
            LOGGER.info("\nbaseline_merged_learner\nMAE="+str(mae)+
                        "; RMSE="+str(rmse))
            mae_list.append(mae)
            rmse_list.append(rmse)
            n_rules_list.append(len(baseline_merged_learner.rule_set))


            # No rule repetition, green
            no_rule_rep_merged_learner_pred = no_rule_rep_merged_learner.predict(test_x)
            mae = round(mean_absolute_error(test_y, no_rule_rep_merged_learner_pred),6)
            rmse = round(sqrt(mean_squared_error(test_y, no_rule_rep_merged_learner_pred)),6)
            LOGGER.info("\nno_rule_rep_merged_learner\nMAE="+str(mae)+
                        "; RMSE="+str(rmse))
            mae_list.append(mae)
            rmse_list.append(rmse)
            n_rules_list.append(len(no_rule_rep_merged_learner.rule_set))

            
            # Best default rule, purple
            best_df_rule_merged_learner_pred = best_df_rule_merged_learner.predict(test_x)
            mae = round(mean_absolute_error(test_y, best_df_rule_merged_learner_pred),6)
            rmse = round(sqrt(mean_squared_error(test_y, best_df_rule_merged_learner_pred)),6)
            LOGGER.info("\nbest_df_rule_merged_learner_pred\nMAE="+str(mae)+
                        "; RMSE="+str(rmse))
            mae_list.append(mae)
            rmse_list.append(rmse)
            n_rules_list.append(len(best_df_rule_merged_learner.rule_set))


            # Best default rule and no rule repetition, red
            best_df_no_rep_merged_learner_pred = best_df_no_rep_merged_learner.predict(test_x)
            mae = round(mean_absolute_error(test_y, best_df_no_rep_merged_learner_pred),6)
            rmse = round(sqrt(mean_squared_error(test_y, best_df_no_rep_merged_learner_pred)),6)
            LOGGER.info("\nbest_df_rule_merged_learner_pred\nMAE="+str(mae)+
                        "; RMSE="+str(rmse))
            mae_list.append(mae)
            rmse_list.append(rmse)
            n_rules_list.append(len(best_df_no_rep_merged_learner.rule_set))


            mae_temp.append(mae_list)
            rmse_temp.append(rmse_list)
            nrules_temp.append(n_rules_list)

        # average of errors of CV
        list_test_mae.append(list(map(mean, zip(*mae_temp))))
        list_test_rmse.append(list(map(mean, zip(*rmse_temp))))
        list_n_rules.append(list(map(mean, zip(*nrules_temp))))



    plt.subplot(1,3,1)
    n = 5
    colors_list = np.array(["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"]*n)
    n_models_list = np.array([2]*n + [4]*n + [8]*n + [16]*n + [32]*n)

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
