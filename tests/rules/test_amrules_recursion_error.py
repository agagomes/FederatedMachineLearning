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

import matplotlib.pyplot as plt

import copy
import time

import logging


LOGGER = logging.getLogger(__name__)


def test_merge_amrules():

    ### split based on attribute, probabilistically, streaming
    ### Node vs merged

    ###############################################################
    # Variables
    USED_CARS_DF = False # True False
    DDOS_DF = True # True False

    ORDERED_RULES = False
    NUMBER_MODELS = [32]#2,4,8,16,32]
    ARTIFICIAL_DF = False
    NOMINAL_ATTRIBUTES = None
    REMOVE_POOR_ATTS = False
    NODE_PROBABILITY = 2/3 # 0 1 1/3 2/3 
    # NODE_PROBABILITY
    #    0 for sequential equally distributed 
    #    ]0,1[ for probabilistically distributed 
    #    1 to distribute att value to a specific node without attention to do it equally
    #ATTRIBUTE_SPLIT_IDX = None 
    seed(123) #random package

    LOGGER.info("\n\nORDERED_RULES="+str(ORDERED_RULES)+
                "\nREMOVE_POOR_ATTS="+str(REMOVE_POOR_ATTS)+
                "\nARTIFICIAL_DF="+str(ARTIFICIAL_DF)+"\n")


    ###############################################################
    # Data
    if not ARTIFICIAL_DF:
        #df = pd.read_csv("/Users/mariajoaolavoura/Desktop/datasets/CartExample/cart_delve.data",
        #                  names=list(map(str, range(0,11))), sep="  ")
        df = None
        if USED_CARS_DF:
            df = pd.read_csv("/Users/mariajoaolavoura/Desktop/datasets/used_cars/used_cars_cleaned.csv")
        elif DDOS_DF:
            #df = pd.read_csv("/Users/mariajoaolavoura/Desktop/datasets/dos/ddos.csv")
            """dtypes={ 'Src IP': 'category',
                    'Flow Duration': 'uint32',
                    'Tot Fwd Pkts': 'uint32',
                    'Tot Bwd Pkts': 'uint32',
                    'Fwd Pkt Len Max': 'float32',
                    'Fwd Pkt Len Min': 'float32',
                    'Bwd Pkt Len Max': 'float32',
                    'Bwd Pkt Len Min': 'float32',
                    'Flow Byts/s': 'float32',
                    'Flow Pkts/s': 'float32',
                    'Flow IAT Mean': 'float32',
                    'Flow IAT Std': 'float32',
                    'Flow IAT Max': 'float32',
                    'Fwd IAT Std': 'float32',
                    'Bwd IAT Mean': 'float32',
                    'Bwd IAT Std': 'float32',
                    'Bwd Pkts/s': 'float32',
                    'Pkt Len Var': 'float32',
                    'SYN Flag Cnt': 'category',
                    'RST Flag Cnt': 'category',
                    'PSH Flag Cnt': 'category',
                    'ACK Flag Cnt': 'category',
                    'CWE Flag Count': 'category',
                    'ECE Flag Cnt': 'category',
                    'Down/Up Ratio': 'float32',
                    'Fwd Byts/b Avg': 'uint32',
                    'Fwd Pkts/b Avg': 'uint32',
                    'Fwd Blk Rate Avg': 'uint32',
                    'Bwd Byts/b Avg': 'uint32',
                    'Bwd Pkts/b Avg': 'uint32',
                    'Bwd Blk Rate Avg': 'uint32',
                    'Init Fwd Win Byts': 'uint32',
                    'Init Bwd Win Byts': 'uint32',
                    'Active Mean': 'float32',
                    'Active Std': 'float32',
                    'Idle Std': 'float32',
                    'Label': 'uint32'
            }"""
            df= pd.read_csv('/Users/mariajoaolavoura/Desktop/datasets/dos/kaggle/split/ddosbaa.csv')#,
                            # dtype=dtypes,
                            # parse_dates=['Timestamp'],
                            # usecols=[*dtypes.keys(), 'Timestamp'],
                            # engine='c',
                            # low_memory=True 
            # )
        
        LOGGER.info("\ndf=\n"+str(df.head()))

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

    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=False)
    
    
    ###############################################################
    # Sort only the train set by attribute
    
    ######
    # Used Cars dataset
    if USED_CARS_DF:
        DECIMAL_NUMBERS = 1
        NOMINAL_ATTRIBUTES = ['region','manufacturer','fuel','title_status','transmission','drive','type','state']
        att = 'state' #region, manufacturer, state
        
        temp = x_train.copy()
        temp['price'] = y_train['price']#.values

        temp = temp.sort_values(att)

        x_train = temp.drop(columns=['price'])
        y_train['price'] = temp['price']#.values
        
        att_list = x_train[att]
        att_unique_list = x_train[att].unique()
        att_split_idx = x_train.columns.get_loc(att)

    ######
    # DDOS
    elif DDOS_DF:
        DECIMAL_NUMBERS = 0
        NOMINAL_ATTRIBUTES = ['SYN Flag Cnt',
                                'RST Flag Cnt',
                                'PSH Flag Cnt',
                                'ACK Flag Cnt',
                                'CWE Flag Count',
                                'ECE Flag Cnt']
        # att = 'Source IP'
        att = 'Src IP'
        att2 = 'Timestamp'
        target = 'Label'
        
        temp = x_train.copy()
        temp[target] = y_train[target]#.values
        
        temp[att] = temp[att].astype('category')
        temp[att2] = pd.to_datetime(temp[att2])

        temp = temp.sort_values([att, att2], ascending=[True, True])
        
        att_list = x_train[att]
        att_unique_list = x_train[att].unique()
        att_split_idx = x_train.columns.get_loc(att)

        x_train = temp.drop(columns=[att,att2, target])
        y_train[target] = temp[target]#.values
    
    
    LOGGER.info("\nDECIMAL_NUMBERS="+str(DECIMAL_NUMBERS)+
                "\natt_unique_list=\n"+str(att_unique_list)+
                "; len="+str(len(att_unique_list)))

    if not ARTIFICIAL_DF:
        X = x_train.values.tolist()
        y = y_train.values.tolist()
        y = [item for sublist in y for item in sublist]
    else:
        X = x_train
        y = y_train

    
    ###############################################################
    #list_training_error = []
    # list_model_states = []
    # list_model_ninstances = []
    # list_model_times = []

    # ###############################################################
    # centralised model
    LOGGER.info("\nGoing to train centralised model")
    centralized = AMRulesRegressor(grace_period=200,
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=ORDERED_RULES,
                                drift_detector=None,
                                remove_poor_atts=REMOVE_POOR_ATTS,
                                nominal_attributes=NOMINAL_ATTRIBUTES,
                                decimal_numbers=DECIMAL_NUMBERS,
                                max_rules=20)
    
    centralized_start = time.time()
    centralized.partial_fit(X, y)
    centralized_end = time.time()
    LOGGER.info("\nbaseline model=\n"+str(centralized.get_model_description())+
                "\ntime="+str(centralized_end - centralized_start))
    #LOGGER.info("\nbaseline model finished training")
    
    """
    if not ARTIFICIAL_DF:
        X = x_test.values.tolist()
        y = y_test.values.tolist()
    else:
        X = x_test
        y = y_test
    """

    ###############################################################
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

        ###############################################################
        # Models
        LOGGER.info("\n\ntraining "+str(n_models)+" models")

        learners = list()
        #learners_df = {}
        for i in range(n_models):
            #learners_df[i] = list()
            learners += [
                AMRulesRegressor(grace_period=200,
                                    tie_threshold=0.05,
                                    expand_confidence=0.01,
                                    ordered_rules=ORDERED_RULES,
                                    drift_detector=None,
                                    remove_poor_atts=REMOVE_POOR_ATTS,
                                    nominal_attributes=NOMINAL_ATTRIBUTES,
                                    decimal_numbers=DECIMAL_NUMBERS,
                                    max_rules=20)
            ]


        ###############################################################
        # store info
        dict_model_attsplit = {}
        dict_model_ninstances = {}
        dict_model_times = {}
        dict_model_nrules = {}
        dict_model_mae = {}
        dict_model_rmse = {}


        ###############################################################
        # Train

        l = int(len(X)/n_models) # dividing instances to keep models leveleds
        model_idx = 0 # starting learner who has biggest probability of receiving a instance
        # start = time.time()
        node_start = time.time()

        # 1 to distribute att value to a specific node without attention to do it equally
        if NODE_PROBABILITY == 1:
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
                    # store node training time
                    node_end = time.time()
                    dict_model_times[model_idx] = node_end-node_start   
                    node_start = time.time()

                    model_idx += 1
                    # LOGGER.info(model_idx)
                    # there are no more learners
                    if model_idx >= n_models:
                        break # stops only this for-loop to train the learners
                # train 
                #learners_df[model_idx] += X_i[ATTRIBUTE_SPLIT_IDX]]  
                learner = learners[model_idx]
                # start = time.time()
                learner.partial_fit(X_i, y_i)
                try:
                    dict_model_attsplit[model_idx].add(att_i)
                    dict_model_ninstances[model_idx] += 1
                    # dict_model_times[model_idx] += [end-start]
                except (TypeError, KeyError):
                    dict_model_attsplit[model_idx] = set()
                    dict_model_attsplit[model_idx].add(att_i)
                    dict_model_ninstances[model_idx] = 1
                    # dict_model_times[model_idx] = [end-start]      

        # ]0,1[ for probabilistically distributed
        elif NODE_PROBABILITY > 0:
            
            for i in range(len(X)):
                X_i = [X[i]]
                y_i = [y[i]]
                # att_i = X_i[0][att_split_idx]
                att_i = att_list[i]

                if uniform(0, 1) <= NODE_PROBABILITY:
                    # change learner who has biggest probability of receiving the instance
                    x_idx = (model_idx+1)*l
                    if i == x_idx:
                        model_idx += 1
                        # there are no more learners
                        if model_idx >= n_models:
                            break # stops only this for-loop to train the learners
                    # train
                    node_start = time.time()    
                    learners[model_idx].partial_fit(X_i, y_i)
                    node_end = time.time()
                    # LOGGER.info("training on model_idx="+str(model_idx))
                    
                    try:
                        dict_model_attsplit[model_idx].add(att_i)
                        dict_model_ninstances[model_idx] += 1
                        dict_model_times[model_idx] += node_end-node_start
                    except (TypeError, KeyError):
                        dict_model_attsplit[model_idx] = set()
                        dict_model_attsplit[model_idx].add(att_i)
                        dict_model_ninstances[model_idx] = 1
                        dict_model_times[model_idx] = node_end-node_start

                else:
                    # train
                    allowed_values = list(range(0, n_models))
                    allowed_values.remove(model_idx)
                    random_model_idx = choice(allowed_values)  
                    
                    node_start = time.time()
                    learners[random_model_idx].partial_fit(X_i, y_i)
                    node_end = time.time()
                    # LOGGER.info("training on random_model_idx="+str(random_model_idx))

                    try:
                        dict_model_attsplit[random_model_idx].add(att_i)
                        dict_model_ninstances[random_model_idx] += 1
                        dict_model_times[random_model_idx] += node_end-node_start
                    except (TypeError, KeyError):
                        dict_model_attsplit[random_model_idx] = set()
                        dict_model_attsplit[random_model_idx].add(att_i)
                        dict_model_ninstances[random_model_idx] = 1
                        dict_model_times[random_model_idx] = node_end-node_start


       
        # 0 for sequential equally distributed 
        else:    
            for i in range(len(X)):
                X_i = [X[i]]
                y_i = [y[i]]
                x_idx = (model_idx+1)*l
                att_i = X_i[0][att_split_idx]
                if i == x_idx:
                    # store node training time
                    node_end = time.time()
                    dict_model_times[model_idx] = node_end-node_start    
                    node_start = time.time()

                    model_idx += 1
                    LOGGER.info("model_idx="+str(model_idx))
                    # there are no more learners
                    if model_idx >= n_models:
                        break # stops only this for-loop to train the learners
                # train 
                #learners_df[model_idx] += X_i[ATTRIBUTE_SPLIT_IDX]]  
                learner = learners[model_idx]
                learner.partial_fit(X_i, y_i)
                # LOGGER.info("att_i="+str(att_i))
                try:
                    dict_model_attsplit[model_idx].add(att_i)
                    dict_model_ninstances[model_idx] += 1
                    # dict_model_times[model_idx] += [end-start]
                except (TypeError, KeyError):
                    dict_model_attsplit[model_idx] = set()
                    dict_model_attsplit[model_idx].add(att_i)
                    dict_model_ninstances[model_idx] = 1
                    # dict_model_times[model_idx] = [end-start]
            
            
        if model_idx == (n_models-1) and (NODE_PROBABILITY == 0 or NODE_PROBABILITY == 1):
            node_end = time.time()
            dict_model_times[model_idx] = [node_end-node_start]

        # LOGGER.info("\n\ndict_model_attsplit="+str(dict_model_attsplit)+
        #                     "\n\ndict_model_ninstances="+str(dict_model_ninstances)+
        #                     "\n\ndict_model_times="+str(dict_model_times))

        # LOGGER.info("\n\ni="+str(i)+"/"+str(len(X)))

        # end = time.time()

        # list_model_states.append(dict_model_attsplit)
        # list_model_ninstances.append(dict_model_ninstances)
        # list_model_times.append(dict_model_times) # list_model_times.append(np.mean(list(dict_model_times.values())))

        # LOGGER.info("\n\ndict_model_attsplit="+str(dict_model_attsplit)+
        #             "\n\ndict_model_ninstances="+str(dict_model_ninstances)+
        #             "\n\ndict_model_times="+str(dict_model_times)+
        #             "\n\nlist(dict_model_times.values())="+str(list(dict_model_times.values()))+
        #             "\n\nnp.mean(list(dict_model_times.values()))="+str(np.mean(list(dict_model_times.values())))+
        #             "\n\nlist_model_states="+str(list_model_states)+
        #             "\n\nlist_model_ninstances="+str(list_model_ninstances)+
        #             "\n\nlist_model_times="+str(list_model_times)+
        #             "\n\n")



        ###############################################################
        # Merge model with best default rule and no rule repetition
        best_df_no_rep_merged_learner = AMRulesRegressor(grace_period=200,
                                    tie_threshold=0.05,
                                    expand_confidence=0.01,
                                    ordered_rules=ORDERED_RULES,
                                    drift_detector=None,
                                    remove_poor_atts=REMOVE_POOR_ATTS,
                                    nominal_attributes=NOMINAL_ATTRIBUTES,
                                    decimal_numbers=DECIMAL_NUMBERS,
                                    max_rules=20)
        

        best_df_no_rep_merged_learner.samples_seen += learners[0].samples_seen 
        best_df_no_rep_merged_learner.sum_target += learners[0].sum_target
        best_df_no_rep_merged_learner.sum_target_square += learners[0].sum_target_square 
        best_df_no_rep_merged_learner.sum_attribute = learners[0].sum_attribute
        best_df_no_rep_merged_learner.sum_attribute_squares = learners[0].sum_attribute_squares
        best_df_no_rep_merged_learner.rule_set += learners[0].rule_set 
        best_df_no_rep_merged_learner.default_rule = copy.deepcopy(learners[0].default_rule)
        best_df_no_rep_merged_learner.n_attributes_df = copy.deepcopy(learners[0].n_attributes_df)

        LOGGER.info("\n\nbaseline_merged_learner node0 =\n"+str(learners[0].get_model_description()))
        LOGGER.info("\ndict_model_attsplit[0]"+str(dict_model_attsplit[0])+"; len="+str(len(dict_model_attsplit[0])))
        LOGGER.info("\ndict_model_ninstances[0]"+str(dict_model_ninstances[0]))
        LOGGER.info("\ndict_model_times[0]"+str(dict_model_times[0]))

        for i in range(1,n_models):
            LOGGER.info("\n\nbaseline_merged_learner node "+str(i)+"=\n"+str(learners[i].get_model_description()))
            LOGGER.info("\ndict_model_attsplit["+str(i)+"]"+str(dict_model_attsplit[i])+"; len="+str(len(dict_model_attsplit[i])))
            LOGGER.info("\ndict_model_ninstances["+str(i)+"]"+str(dict_model_ninstances[i]))
            LOGGER.info("\ndict_model_times["+str(i)+"]"+str(dict_model_times[i]))
            # learner = copy.deepcopy(learners[i])
            learner = learners[i]
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
        #LOGGER.info("\n\nbest_df_no_rep_merged_learner model finished merge")


        ###############################################################
        # Predict

        if not ARTIFICIAL_DF:
            # ddos dataset
            if DDOS_DF:
                x_test = x_test.drop(columns=[att,att2])
            
            X = x_test.values.tolist()
            y = y_test.values.tolist()
        else:
            X = x_test
            y = y_test


        LOGGER.info("\n\nstarted predicting and calculating errors")


        # Centralized, blue
        centralized_pred = centralized.predict(X) #blue
        # LOGGER.info("\nbaseline\nMAE="+str(baseline_mae)+
        #             "; RMSE="+str(baseline_rmse))

        dict_model_attsplit["centralized"] = set()
        dict_model_ninstances["centralized"] = x_train.shape[0]
        dict_model_times["centralized"] = centralized_end-centralized_start
        dict_model_nrules["centralized"] = len(centralized.rule_set)
        dict_model_mae["centralized"] = round(mean_absolute_error(y, centralized_pred),2)
        dict_model_rmse["centralized"] = round(sqrt(mean_squared_error(y, centralized_pred)),2)


        # Nodes, green
        for i in range(n_models):
            node_pred = learners[i].predict(X)

            dict_model_nrules[i] = len(learners[i].rule_set)
            dict_model_mae[i] = round(mean_absolute_error(y, node_pred),2)
            dict_model_rmse[i] = round(sqrt(mean_squared_error(y, node_pred)),2)
            # LOGGER.info("\nlearner"+str(i)+"\nMAE="+str(mae)+
            #             "; RMSE="+str(rmse))



        # Best default rule and no rule repetition, red
        best_df_no_rep_merged_learner_pred = best_df_no_rep_merged_learner.predict(X)
        # LOGGER.info("\nbest_df_rule_merged_learner_pred\nMAE="+str(mae)+
        #             "; RMSE="+str(rmse))

        dict_model_attsplit["merged"] = set()
        dict_model_ninstances["merged"] = sum(list(dict_model_ninstances.values())[1:]) # sum of nodes values
        dict_model_times["merged"] = sum(list(dict_model_times.values())[1:]) # sum of nodes values
        dict_model_nrules["merged"] = len(best_df_no_rep_merged_learner.rule_set)
        dict_model_mae["merged"] = round(mean_absolute_error(y, best_df_no_rep_merged_learner_pred),2)
        dict_model_rmse["merged"] = round(sqrt(mean_squared_error(y, best_df_no_rep_merged_learner_pred)),2)

        # list_test_mae.append(copy.deepcopy(mae_list))
        # list_test_rmse.append(copy.deepcopy(rmse_list))
        # list_n_rules.append(copy.deepcopy(n_rules_list))



        ###############################################################
        # Graphs

        LOGGER.info("\n\ndict_model_attsplit="+str(dict_model_attsplit)+
                    "\n\ndict_model_ninstances="+str(dict_model_ninstances)+
                    "\n\ndict_model_times="+str(dict_model_times)+
                    "\n\ndict_model_nrules="+str(dict_model_nrules)+
                    "\n\ndict_model_mae="+str(dict_model_mae)+
                    "\n\ndict_model_rmse="+str(dict_model_rmse))


        colors_list = ["tab:blue"] # centralized
        colors_list += ["tab:green"]*n_models # nodes
        colors_list += ["tab:red"] # merged
        colors_list = np.array(colors_list)

        models_list = ["centralized"]
        models_list += ["node"+str(i) for i in range(n_models)]
        models_list += ["merged"]
        models_list = np.array(models_list)

        
        plt.figure()
        plt.subplot(2,3,1)
        attsplit = [len(s) for s in list(dict_model_attsplit.values())]# rules_l = sum(list_n_rules, []) # np.array(list_n_rules).ravel()
        max_lim = int(max(attsplit) + max(attsplit)/3)
        plt.ylim([0,max_lim])
        
        plt.scatter(models_list, attsplit, c=colors_list)
        plt.xlabel('Models')
        plt.ylabel('Number of unique values of split attribute')
        plt.title('Number of unique values of split attribute per node')
        plt.tight_layout()

        
        plt.subplot(2,3,2)
        ninstances = list(dict_model_ninstances.values()) # mae_l = sum(list_test_mae, []) # np.array(list_test_mae).ravel()
        max_lim = int(max(ninstances) + max(ninstances)/3)
        plt.ylim([0,max_lim])
        
        plt.scatter(models_list, ninstances, c=colors_list)
        plt.xlabel('Models')
        plt.ylabel('Number of instances')
        plt.title('Number of instances across models')
        plt.tight_layout()


        plt.subplot(2,3,3)
        nrules = list(dict_model_nrules.values()) # rules_l = sum(list_n_rules, []) # np.array(list_n_rules).ravel()
        max_lim = int(max(nrules) + max(nrules)/3)
        plt.ylim([0,max_lim])
        
        plt.scatter(models_list, nrules, c=colors_list)
        plt.xlabel('Models')
        plt.ylabel('Number of rules')
        plt.title('Number of rules across models')
        plt.tight_layout()


        plt.subplot(2,3,4)
        times = list(dict_model_times.values()) # mae_l = sum(list_test_mae, []) # np.array(list_test_mae).ravel()
        max_lim = int(max(times) + max(times)/3)
        plt.ylim([0,max_lim])
        
        plt.scatter(models_list, times, c=colors_list)
        plt.xlabel('Models')
        plt.ylabel('Time')
        plt.title('Train time across models')
        plt.tight_layout()
        

        plt.subplot(2,3,5)
        mae = list(dict_model_mae.values()) # mae_l = sum(list_test_mae, []) # np.array(list_test_mae).ravel()
        max_lim = int(max(mae) + max(mae)/3)
        plt.ylim([0,max_lim])
        
        plt.scatter(models_list, mae, c=colors_list)
        plt.xlabel('Models')
        plt.ylabel('MAE')
        plt.title('Test MAE across models')
        plt.tight_layout()


        plt.subplot(2,3,6)
        rmse = list(dict_model_rmse.values()) # mae_l = sum(list_test_mae, []) # np.array(list_test_mae).ravel()
        max_lim = int(max(rmse) + max(rmse)/3)
        plt.ylim([0,max_lim])
        
        plt.scatter(models_list, rmse, c=colors_list)
        plt.xlabel('Models')
        plt.ylabel('RMSE')
        plt.title('Test RMSE across models')
        plt.tight_layout()

        plt.show()

test_merge_amrules()