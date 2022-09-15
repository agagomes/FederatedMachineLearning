from skmultiflow.rules import AMRulesRegressor
from skmultiflow.drift_detection import PageHinkley
import numpy as np
import pandas as pd
import sklearn.datasets as dt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from statistics import mode
from random import uniform, choice, seed
from operator import itemgetter

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
    # RULE_PREDICTION = 'first_hit' # first_hit weighted_max
    NUMBER_MODELS = [2,4,6,8]
    #NUMBER_MODELS = [2,4,6,8,10,12,14,16,32]
    ARTIFICIAL_DF = False
    NOMINAL_ATTRIBUTES = None
    NODE_PROBABILITY = 2/3 # 0 1 1/3 2/3 
    # NODE_PROBABILITY
    #    0 for sequential equally distributed 
    #    ]0,1[ for probabilistically distributed 
    #    1 to distribute att value to a specific node without attention to do it equally
    #ATTRIBUTE_SPLIT_IDX = None 
    seed(123) #random package

    LOGGER.info("\n\nORDERED_RULES="+str(ORDERED_RULES)+
                # "\nRULE_PREDICTION="+str(RULE_PREDICTION)+
                "\nARTIFICIAL_DF="+str(ARTIFICIAL_DF)+"\n")

    #####################
    # Data
    if not ARTIFICIAL_DF:
        df = pd.read_csv("/home/angelo/Desktop/databases/diamondsOrganized.csv")
        #df = pd.read_csv("/home/angelo/Desktop/databases/used_cars_cleaned.csv")                  
        #df = pd.read_csv("/Users/mariajoaolavoura/Desktop/datasets/CartExample/cart_delve.data",
        #                  names=list(map(str, range(0,11))), sep="  ")

        LOGGER.info("\ndf=\n"+str(df.head()))

        x = df.iloc[:,:-1] #atributos
        y = df.iloc[:,-1:] #target_value (price)

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

    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=False)
    

    #####################
    # Sort only the train set
    #used cars 
    """
    NOMINAL_ATTRIBUTES = ['region','manufacturer','fuel','title_status','transmission','drive','type','state']
    att = 'state'
    """
    #diamonds
    NOMINAL_ATTRIBUTES = ['carat','cut','color','clarity','depth','table','x.length','y.length','z.length']
    att = 'clarity'

    temp = x_train.copy()
    temp['price'] = y_train['price'].values
    temp = temp.sort_values(att) #df = df.sort_values('10') # sort by target
    x_train = temp.drop(columns=['price'])
    y_train['price'] = temp['price'].values
    att_unique_list = x_train[att].unique()
    att_split_idx = x_train.columns.get_loc(att)
    #LOGGER.info("att_unique_list="+str(att_unique_list))

    if not ARTIFICIAL_DF:
        X = x_train.values.tolist()
        y = y_train.values.tolist()
        y = [item for sublist in y for item in sublist]
    else:
        X = x_train
        y = y_train

    
    #####################
    #list_training_error = []
    list_test_mae = [] # [[2n_baseline, 2n_no_rep, 2n_best_df, 2n_best_df_no_rep], [4n_baseline, ...],[...],[...],[...]]
    list_test_rmse = []
    list_n_rules = []
    list_model_states = []
    list_model_ninstances = []
    list_model_times = []

    # #####################
    # centralised model
    base_line = AMRulesRegressor(grace_period=200,
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=ORDERED_RULES,
                                # rule_prediction=RULE_PREDICTION,
                                drift_detector=None,
                                nominal_attributes=NOMINAL_ATTRIBUTES,
                                decimal_numbers=1,
                                max_rules=20)

    start = time.time()
    base_line.partial_fit(X, y)
    end = time.time()
    #LOGGER.info("\nbaseline model=\n"+str(base_line.get_model_description())+
    #            "\ntime="+str(end - start))
    LOGGER.info("\nbaseline model finished training")

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

        # LOGGER.info(list(map(itemgetter(att_split_idx), X)))

        #####################
        # Models
        LOGGER.info("\n\ntraining "+str(n_models)+" models")

        learners = list()
        #learners_df = {}
        for i in range(n_models):
            LOGGER.info("\n1")
            #learners_df[i] = list()
            learners += [
                AMRulesRegressor(grace_period=200,
                                    tie_threshold=0.05,
                                    expand_confidence=0.01,
                                    ordered_rules=ORDERED_RULES,
                                    # rule_prediction=RULE_PREDICTION,
                                    drift_detector=None,
                                    nominal_attributes=NOMINAL_ATTRIBUTES,
                                    decimal_numbers=1,
                                    max_rules=20)
            ]


        #####################
        # Train
        dict_model_states = {}
        dict_model_ninstances = {}
        dict_model_times = {}
        LOGGER.info("\n2")
        l = int(len(X)/n_models) # dividing instances to keep models leveled
        l = int(len(X)/n_models) # dividing instances to keep models leveled
        LOGGER.info("\n3")
        model_idx = 0 # starting learner who has biggest probability of receiving a instance
        LOGGER.info("\n4")
        start = time.time()

        # 1 to distribute att value to a specific node without attention to do it equally
        if NODE_PROBABILITY == 1:
            LOGGER.info("\n5")
            # dividing instances to guarantee 100% of that att-value is in 1 node 
            l = int(len(att_unique_list)/n_models) 
            for i in range(len(X)):
                X_i = [X[i]]
                y_i = [y[i]]
                
                att_i = X_i[0][att_split_idx]
                att_values = att_unique_list[model_idx*l : (model_idx+1)*l]
                #LOGGER.info("\n\nmodel_idx="+str(model_idx)+\
                #             "\nlen(att_unique_list)="+str(len(att_unique_list))+\
                #             "\natt_i="+str(att_i)+\
                #             "\natt_values idx=["+str(model_idx*l)+","+str((model_idx+1)*l)+"]"+\
                #             "\natt_values="+str(att_values)+\
                #             "\natt_i not in att_values="+str((att_i not in att_values)))

                if att_i not in att_values: # works well because X is ordered by att
                    model_idx += 1
                    # LOGGER.info(model_idx)
                    # there are no more learners
                    if model_idx >= n_models:
                        # LOGGER.info("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB!")
                        break # stops only this for-loop to train the learners
                # train 
                #learners_df[model_idx] += X_i[ATTRIBUTE_SPLIT_IDX]]  
                learner = learners[model_idx]
                start = time.time()
                learner.partial_fit(X_i, y_i)
                try:
                    dict_model_states[model_idx].add(att_i)
                    dict_model_ninstances[model_idx] += 1
                    dict_model_times[model_idx] += [end-start]
                except (TypeError, KeyError):
                    dict_model_states[model_idx] = set()
                    dict_model_states[model_idx].add(att_i)
                    dict_model_ninstances[model_idx] = 1
                    dict_model_times[model_idx] = [end-start]      

        # ]0,1[ for probabilistically distributed
        elif NODE_PROBABILITY > 0:
            LOGGER.info("\n6")
            for i in range(len(X)):
                X_i = [X[i]]
                y_i = [y[i]]
                att_i = X[0][att_split_idx]
                if uniform(0, 1) <= NODE_PROBABILITY:
                    # change learner who has biggest probability of receiving the instance
                    x_idx = (model_idx+1)*l
                    if i == x_idx:
                        model_idx += 1
                        # LOGGER.info(model_idx)
                        # there are no more learners
                        if model_idx >= n_models:
                            break # stops only this for-loop to train the learners
                    # train    
                    #learners_df[model_idx] += [X_i[ATTRIBUTE_SPLIT_IDX]]
                    learners[model_idx].partial_fit(X_i, y_i)
                else:
                    # train
                    allowed_values = list(range(0, n_models))
                    allowed_values.remove(model_idx)
                    random_model_idx = choice(allowed_values)  
                    #learners_df[random_model_idx] += X_i[ATTRIBUTE_SPLIT_IDX]]
                    learners[random_model_idx].partial_fit(X_i, y_i)
                try:
                    dict_model_states[model_idx].add(att_i)
                    dict_model_ninstances[model_idx] += 1
                    dict_model_times[model_idx] += [end-start]
                except (TypeError, KeyError):
                    dict_model_states[model_idx] = set()
                    dict_model_states[model_idx].add(att_i)
                    dict_model_ninstances[model_idx] = 1
                    dict_model_times[model_idx] = [end-start]   
       
        # 0 for sequential equally distributed 
        else:    
            LOGGER.info("\n7")
            for i in range(len(X)):
                X_i = [X[i]]
                y_i = [y[i]]
                x_idx = (model_idx+1)*l
                att_i = X[0][att_split_idx]
                if i == x_idx:
                    model_idx += 1
                    # LOGGER.info("model_idx="+str(model_idx))
                    # there are no more learners
                    if model_idx >= n_models:
                        break # stops only this for-loop to train the learners
                # train 
                #learners_df[model_idx] += X_i[ATTRIBUTE_SPLIT_IDX]]  
                learner = learners[model_idx]
                learner.partial_fit(X_i, y_i)
                # LOGGER.info("att_i="+str(att_i))
                try:
                    dict_model_states[model_idx].add(att_i)
                    dict_model_ninstances[model_idx] += 1
                    dict_model_times[model_idx] += [end-start]
                except (TypeError, KeyError):
                    dict_model_states[model_idx] = set()
                    dict_model_states[model_idx].add(att_i)
                    dict_model_ninstances[model_idx] = 1
                    dict_model_times[model_idx] = [end-start]   

        # LOGGER.info("\n\ni="+str(i)+"/"+str(len(X)))
        LOGGER.info("\n8")
        end = time.time()

        list_model_states.append(dict_model_states)
        list_model_ninstances.append(dict_model_ninstances)
        list_model_times.append(dict_model_times) # list_model_times.append(np.mean(list(dict_model_times.values())))

        # LOGGER.info("\n\ndict_model_states="+str(dict_model_states)+
        #             "\n\ndict_model_ninstances="+str(dict_model_ninstances)+
        #             "\n\ndict_model_times="+str(dict_model_times)+
        #             "\n\nlist(dict_model_times.values())="+str(list(dict_model_times.values()))+
        #             "\n\nnp.mean(list(dict_model_times.values()))="+str(np.mean(list(dict_model_times.values())))+
        #             "\n\nlist_model_states="+str(list_model_states)+
        #             "\n\nlist_model_ninstances="+str(list_model_ninstances)+
        #             "\n\nlist_model_times="+str(list_model_times)+
        #             "\n\n")
                            
        #####################
        # Merge models

        # Baseline
        LOGGER.info("\n9")
        baseline_merged_learner = AMRulesRegressor(grace_period=200,
                                    tie_threshold=0.05,
                                    expand_confidence=0.01,
                                    ordered_rules=ORDERED_RULES,
                                    # rule_prediction=RULE_PREDICTION,
                                    drift_detector=None,
                                    nominal_attributes=NOMINAL_ATTRIBUTES,
                                    decimal_numbers=1,
                                    max_rules=20)
        LOGGER.info("\n10")

        # baseline_merged_learner.samples_seen += copy.deepcopy(learners[0].samples_seen)
        # baseline_merged_learner.sum_target += copy.deepcopy(learners[0].sum_target)
        # baseline_merged_learner.sum_target_square += copy.deepcopy(learners[0].sum_target_square)
        # baseline_merged_learner.sum_attribute = copy.deepcopy(learners[0].sum_attribute)
        # baseline_merged_learner.sum_attribute_squares = copy.deepcopy(learners[0].sum_attribute_squares)
        # baseline_merged_learner.rule_set += copy.deepcopy(learners[0].rule_set)
        # baseline_merged_learner.default_rule = copy.deepcopy(learners[0].default_rule)
        # baseline_merged_learner.n_attributes_df = copy.deepcopy(learners[0].n_attributes_df)
        baseline_merged_learner.samples_seen += learners[0].samples_seen
        baseline_merged_learner.sum_target += learners[0].sum_target
        baseline_merged_learner.sum_target_square += learners[0].sum_target_square
        baseline_merged_learner.sum_attribute = learners[0].sum_attribute
        baseline_merged_learner.sum_attribute_squares = learners[0].sum_attribute_squares
        baseline_merged_learner.rule_set += learners[0].rule_set
        baseline_merged_learner.default_rule = learners[0].default_rule
        baseline_merged_learner.n_attributes_df = learners[0].n_attributes_df
        LOGGER.info("\n11")
        for i in range(1,n_models):
            LOGGER.info("\n12")
            # LOGGER.info("\n\nbaseline_merged_learner node"+str(i)+"=\n"+str(learners[i].get_model_description()))
            learner = copy.deepcopy(learners[i])
            LOGGER.info("\n13")
            # learner = learners[i]
            LOGGER.info("learner")
            # for rule in learner.rule_set:SS
            for rule in learner.rule_set:
                baseline_merged_learner.rule_set += [copy.deepcopy(rule)]
                LOGGER.info("rule_set")
                baseline_merged_learner.samples_seen += learner.samples_seen
                # LOGGER.info("samples_seen")
                baseline_merged_learner.sum_target += learner.sum_target
                # LOGGER.info("sum_target")
                baseline_merged_learner.sum_target_square += learner.sum_target_square
                # LOGGER.info("sum_target_square")
                baseline_merged_learner.sum_attribute = [x + y for x, y in zip(baseline_merged_learner.sum_attribute, learner.sum_attribute)] 
                # LOGGER.info("sum_attribute")
                baseline_merged_learner.sum_attribute_squares = [x + y for x, y in zip(baseline_merged_learner.sum_attribute_squares, \
                                                                            learner.sum_attribute_squares)] 
                # LOGGER.info("sum_attribute_squares")
                LOGGER.info("end for")

        #LOGGER.info("\n\nbaseline_merged_learner model=\n"+str(baseline_merged_learner.get_model_description()))
        LOGGER.info("\n\nbaseline_merged_learner model finished merge")

        # No rule repetition
        no_rule_rep_merged_learner = AMRulesRegressor(grace_period=200,
                                    tie_threshold=0.05,
                                    expand_confidence=0.01,
                                    ordered_rules=ORDERED_RULES,
                                    # rule_prediction=RULE_PREDICTION,
                                    drift_detector=None,
                                    nominal_attributes=NOMINAL_ATTRIBUTES,
                                    decimal_numbers=1,
                                    max_rules=20)
        

        no_rule_rep_merged_learner.samples_seen += learners[0].samples_seen
        no_rule_rep_merged_learner.sum_target += learners[0].sum_target
        no_rule_rep_merged_learner.sum_target_square += learners[0].sum_target_square
        no_rule_rep_merged_learner.sum_attribute = learners[0].sum_attribute
        no_rule_rep_merged_learner.sum_attribute_squares = learners[0].sum_attribute_squares
        no_rule_rep_merged_learner.rule_set += learners[0].rule_set
        no_rule_rep_merged_learner.default_rule = learners[0].default_rule
        no_rule_rep_merged_learner.n_attributes_df = learners[0].n_attributes_df

        # no_rule_rep_merged_learner.samples_seen += copy.deepcopy(learners[0].samples_seen)
        # no_rule_rep_merged_learner.sum_target += copy.deepcopy(learners[0].sum_target)
        # no_rule_rep_merged_learner.sum_target_square += copy.deepcopy(learners[0].sum_target_square)
        # no_rule_rep_merged_learner.sum_attribute = copy.deepcopy(learners[0].sum_attribute)
        # no_rule_rep_merged_learner.sum_attribute_squares = copy.deepcopy(learners[0].sum_attribute_squares)
        # no_rule_rep_merged_learner.rule_set += copy.deepcopy(learners[0].rule_set)
        # no_rule_rep_merged_learner.default_rule = copy.deepcopy(learners[0].default_rule)
        # no_rule_rep_merged_learner.n_attributes_df = copy.deepcopy(learners[0].n_attributes_df)

        for i in range(1,n_models):
            learner = copy.deepcopy(learners[i])
            # learner = learners[i]
            # for rule in learner.rule_set:
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
        LOGGER.info("\n\nno_rule_rep_merged_learner model finished merge")


        # Best default rule
        best_df_rule_merged_learner = AMRulesRegressor(grace_period=200,
                                    tie_threshold=0.05,
                                    expand_confidence=0.01,
                                    ordered_rules=ORDERED_RULES,
                                    # rule_prediction=RULE_PREDICTION,
                                    drift_detector=None,
                                    nominal_attributes=NOMINAL_ATTRIBUTES,
                                    decimal_numbers=1,
                                    max_rules=20)
        

        best_df_rule_merged_learner.samples_seen += learners[0].samples_seen
        best_df_rule_merged_learner.sum_target += learners[0].sum_target
        best_df_rule_merged_learner.sum_target_square += learners[0].sum_target_square
        best_df_rule_merged_learner.sum_attribute = learners[0].sum_attribute
        best_df_rule_merged_learner.sum_attribute_squares = learners[0].sum_attribute_squares
        best_df_rule_merged_learner.rule_set += learners[0].rule_set
        best_df_rule_merged_learner.default_rule = learners[0].default_rule
        best_df_rule_merged_learner.n_attributes_df = learners[0].n_attributes_df

        # best_df_rule_merged_learner.samples_seen += copy.deepcopy(learners[0].samples_seen)
        # best_df_rule_merged_learner.sum_target += copy.deepcopy(learners[0].sum_target)
        # best_df_rule_merged_learner.sum_target_square += copy.deepcopy(learners[0].sum_target_square)
        # best_df_rule_merged_learner.sum_attribute = copy.deepcopy(learners[0].sum_attribute)
        # best_df_rule_merged_learner.sum_attribute_squares = copy.deepcopy(learners[0].sum_attribute_squares)
        # best_df_rule_merged_learner.rule_set += copy.deepcopy(learners[0].rule_set)
        # best_df_rule_merged_learner.default_rule = copy.deepcopy(learners[0].default_rule)
        # best_df_rule_merged_learner.n_attributes_df = copy.deepcopy(learners[0].n_attributes_df)

        for i in range(1,n_models):
            learner = copy.deepcopy(learners[i])
            # learner = learners[i]
            # for rule in learner.rule_set:
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

        #LOGGER.info("\n\nbest_df_rule_merged_learner model=\n"+str(best_df_rule_merged_learner.get_model_description()))
        LOGGER.info("\n\nbest_df_rule_merged_learner model finished merge")

                                                                        

        # Best default rule and no rule repetition
        best_df_no_rep_merged_learner = AMRulesRegressor(grace_period=200,
                                    tie_threshold=0.05,
                                    expand_confidence=0.01,
                                    ordered_rules=ORDERED_RULES,
                                    # rule_prediction=RULE_PREDICTION,
                                    drift_detector=None,
                                    nominal_attributes=NOMINAL_ATTRIBUTES,
                                    decimal_numbers=1,
                                    max_rules=20)
        

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
            # learner = learners[i]
            # for rule in learner.rule_set:
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
        LOGGER.info("\n\nbest_df_no_rep_merged_learner model finished merge")


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


        # Nodes, brown
        LOGGER.info("\n\nstarted predicting and calculating errors")
        for i in range(n_models):
            node_pred = learners[i].predict(X)
            mae = round(mean_absolute_error(y, node_pred),6)
            rmse = round(sqrt(mean_squared_error(y, node_pred)),6)
            # LOGGER.info("\nlearner"+str(i)+"\nMAE="+str(mae)+
            #             "; RMSE="+str(rmse))
            mae_list.append(mae)
            rmse_list.append(rmse)
            n_rules_list.append(len(learners[i].rule_set))
        
        # LOGGER.info("\n\nn_rules_list learners\n"+str(n_rules_list))


        # Centralized, blue
        # LOGGER.info("\nbaseline\nMAE="+str(baseline_mae)+
        #             "; RMSE="+str(baseline_rmse))
        mae_list.append(baseline_mae)
        rmse_list.append(baseline_rmse)
        n_rules_list.append(baseline_rule_set_len)

        
        # Baseline, orange
        baseline_merged_learner_pred = baseline_merged_learner.predict(X)
        mae = round(mean_absolute_error(y, baseline_merged_learner_pred),6)
        rmse = round(sqrt(mean_squared_error(y, baseline_merged_learner_pred)),6)
        # LOGGER.info("\nbaseline_merged_learner\nMAE="+str(mae)+
        #             "; RMSE="+str(rmse))
        mae_list.append(mae)
        rmse_list.append(rmse)
        n_rules_list.append(len(baseline_merged_learner.rule_set))


        # No rule repetition, green
        no_rule_rep_merged_learner_pred = no_rule_rep_merged_learner.predict(X)
        mae = round(mean_absolute_error(y, no_rule_rep_merged_learner_pred),6)
        rmse = round(sqrt(mean_squared_error(y, no_rule_rep_merged_learner_pred)),6)
        # LOGGER.info("\nno_rule_rep_merged_learner\nMAE="+str(mae)+
        #             "; RMSE="+str(rmse))
        mae_list.append(mae)
        rmse_list.append(rmse)
        n_rules_list.append(len(no_rule_rep_merged_learner.rule_set))

        
        # Best default rule, purple
        best_df_rule_merged_learner_pred = best_df_rule_merged_learner.predict(X)
        mae = round(mean_absolute_error(y, best_df_rule_merged_learner_pred),6)
        rmse = round(sqrt(mean_squared_error(y, best_df_rule_merged_learner_pred)),6)
        # LOGGER.info("\nbest_df_rule_merged_learner_pred\nMAE="+str(mae)+
        #             "; RMSE="+str(rmse))
        mae_list.append(mae)
        rmse_list.append(rmse)
        n_rules_list.append(len(best_df_rule_merged_learner.rule_set))


        # Best default rule and no rule repetition, red
        best_df_no_rep_merged_learner_pred = best_df_no_rep_merged_learner.predict(X)
        mae = round(mean_absolute_error(y, best_df_no_rep_merged_learner_pred),6)
        rmse = round(sqrt(mean_squared_error(y, best_df_no_rep_merged_learner_pred)),6)
        # LOGGER.info("\nbest_df_rule_merged_learner_pred\nMAE="+str(mae)+
        #             "; RMSE="+str(rmse))
        mae_list.append(mae)
        rmse_list.append(rmse)
        n_rules_list.append(len(best_df_no_rep_merged_learner.rule_set))

        list_test_mae.append(copy.deepcopy(mae_list))
        list_test_rmse.append(copy.deepcopy(rmse_list))
        list_n_rules.append(copy.deepcopy(n_rules_list))


    """
    colors_list = []
    n_models_list = []
    c = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"] # ["tab:blue", "tab:orange", "tab:red"] # 
    for i in range(len(NUMBER_MODELS)):
        colors_list += c[i]*NUMBER_MODELS[i] #["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"]
        n_models_list += [NUMBER_MODELS[i]]*NUMBER_MODELS[i] #5 = len(["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"])
    colors_list = np.array(colors_list)
    n_models_list = np.array(n_models_list)
    """
    k=len(NUMBER_MODELS)
    n = 5
    colors_list = np.array(["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"]*k)
    n_models_list = np.array([2]*n + [4]*n + [6]*n + [8]*n)


    # model states
    # remove duplicates
    model_states = [list(model_states_dict[model_idx])\
                                 for model_states_dict in list_model_states\
                                    for model_idx in sorted(model_states_dict)]
    model_states = sum(model_states, [])


    # colors_list_model_states = []
    # n_models_list_model_states = []
    # c = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"] # ["tab:blue", "tab:orange", "tab:red"] # 
    # for i in range(len(NUMBER_MODELS)):
    #     colors_list += c[i]*NUMBER_MODELS[i] #["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"]
    #     n_models_list += [NUMBER_MODELS[i]]*NUMBER_MODELS[i] #5 = len(["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"])
    # colors_list = np.array(colors_list)
    # n_models_list = np.array(n_models_list)



    model_ninstances = [model_ninstances_dict[model_idx]\
                                    for model_ninstances_dict in list_model_ninstances\
                                        for model_idx in sorted(model_ninstances_dict)]
    #model_ninstances = sum(model_ninstances, [])
    model_times = [model_time_dict[model_idx]\
                                for model_time_dict in list_model_times\
                                    for model_idx in sorted(model_time_dict)]
    model_times = sum(model_times, [])
        
    # LOGGER.info("\nGRAPHS\n\n"+
    #             "\nlen n_models_list="+str(len(n_models_list))+
    #             "\nlen colors_list="+str(len(colors_list))+
    #             "\nlen model_states="+str(len(model_states))+
    #             "\n model_states="+str(model_states)+
    #             "\nlen model_ninstances="+str(len(model_ninstances))+
    #             "\nlen model_times="+str(len(model_times))
    # )

    # LOGGER.info("\nGRAPHS\n\n"+
    #             "\n\nn_models_list="+str(n_models_list)+
    #             #"\n\nlist_model_states="+str(list_model_states)+
    #             "\n\nlist_model_ninstances="+str(list_model_ninstances)+
    #             "\n\nmodel_ninstances="+str(model_ninstances))


    # LOGGER.info("\nGRAPHS\n\ncolors_list="+str(colors_list)+
    #             "\n\nn_models_list="+str(n_models_list)+
    #             "\n\ncolors_list="+str(colors_list)+
    #             "\n\nlist_model_states="+str(list_model_states)+
    #             "\n\nlist_model_ninstances="+str(list_model_ninstances)+
    #             "\n\nlist_model_times="+str(list_model_times))
    # LOGGER.info("\n\nmodel_states="+str(model_states)+
    #             "\n\nmodel_ninstances="+str(model_ninstances)+
    #             "\n\nmodel_times="+str(model_times)+
                # "\n\n")

    """
    plt.figure()
    plt.subplot(1,3,1)
    #list_model_states.append(dict_model_states)
    #list_model_ninstances.append(dict_model_ninstances)
    #model_states = np.array([list(d[k]) for d in list_model_states for k in sorted(d)]).ravel()
    max_lim = int(max(model_states) + max(model_states)/3)
    plt.ylim([0,max_lim])
    plt.scatter(n_models_list, model_states)#, c=colors_list)
    plt.xlabel('Number of nodes')
    plt.ylabel('Region')
    plt.title('Regions in each node')
    plt.tight_layout()

    plt.subplot(1,3,2)
    #model_ninstances = np.array([list(d[k]) for d in list_model_ninstances for k in sorted(d)]).ravel()
    max_lim = int(max(model_ninstances) + max(model_ninstances)/3)
    plt.ylim([0,max_lim])
    plt.scatter(n_models_list, model_ninstances)#, c=colors_list)
    plt.xlabel('Number of nodes')
    plt.ylabel('Number of instances')
    plt.title('Number of instances seen in each node')
    plt.tight_layout()

    plt.show()
    """
    # plt.subplot(1,3,3)
    # #model_times = np.array([list(d[k]) for d in list_model_times for k in sorted(d)]).ravel()
    # max_lim = int(max(model_times) + max(model_times)/3)
    # plt.ylim([0,max_lim])
    # plt.scatter(n_models_list, model_times)#, c=colors_list)
    # plt.xlabel('Number of nodes')
    # plt.ylabel('Number of instances')
    # plt.title('Number of instances seen in each node')
    # plt.tight_layout()


    # colors_list = []
    # n_models_list = []
    # c = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"] # ["tab:blue", "tab:orange", "tab:red"] # 
    # for i in range(len(NUMBER_MODELS)):
    #     colors_list += ["tab:brown"]*NUMBER_MODELS[i] + c #["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"]
    #     n_models_list += [NUMBER_MODELS[i]]*(NUMBER_MODELS[i]+len(c)) #5 = len(["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"])
    # colors_list = np.array(colors_list)
    # n_models_list = np.array(n_models_list)

    plt.figure()
    plt.subplot(1,3,1)
    mae_l = sum(list_test_mae, []) 
    max_lim = int(max(mae_l) + max(mae_l)/3)
    plt.ylim([0,max_lim])
    print(len(n_models_list))
    print(len(mae_l))
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