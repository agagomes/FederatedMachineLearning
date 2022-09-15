from pickle import TRUE
from skmultiflow.rules import AMRulesClassifier, VeryFastDecisionRulesClassifier
from skmultiflow.drift_detection import PageHinkley, DDM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from math import sqrt
import numpy as np
import pandas as pd
from array import array
from statistics import mode
from random import uniform, choice, seed

import matplotlib.pyplot as plt

import copy
import time

import logging
logging.basicConfig(filename='logsinfo.log', filemode='w', level=logging.INFO)
LOGGER = logging.getLogger(__name__)

def test_merge_amrules():


    #df = pd.read_csv("/home/angelo/Desktop/databases/adults.csv")
    df = pd.read_csv("/home/angelo/Desktop/databases/heart.csv")
    #df = pd.read_csv("/home/angelo/Desktop/databases/wine.csv") 
    #df = pd.read_csv("/home/angelo/Desktop/databases/stroke.csv")

    #####################
    # Variables
    AVERAGE='weighted'
    ORDERED_RULES = True
    #NUMBER_MODELS = [2]
    NUMBER_MODELS = [2,4,6]
    NOMINAL_ATTRIBUTES = None
    NODE_PROBABILITY = 1
    # NODE_PROBABILITY
    #    0 for sequential equally distributed 
    #    ]0,1[ for probabilistically distributed 
    #    1 to distribute att value to a specific node without attention to do it equally
    seed(123) #random package

    LOGGER.info("\n\nORDERED_RULES="+str(ORDERED_RULES)+
                "\nNODE_PROBABILITY="+str(NODE_PROBABILITY)+"\n")
    

    #####ADULTS##########################
    #df = df.sort_values("salary")

    
    ######HEART##########################
    df = df.sort_values("HeartDisease")

    
    #####WINE############################
    #df = df.sort_values("quality")

    
    #####STROKE##########################
    #df = df.sort_values("stroke")
    
 


    
    x = df.iloc[:,:-1]
    y = df.iloc[:,-1:]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    #list_training_error = []
    list_test_acc = [] # [[2n_baseline, 2n_no_rep, 2n_best_df, 2n_best_df_no_rep], [4n_baseline, ...],[...],[...],[...]]
    list_test_pre = []
    list_test_rec = []
    list_test_f1 = []
    list_test_auc = []
    list_n_rules = []

    
    X = x_train.values.tolist()
    y = y_train.values.tolist()
    y = [item for sublist in y for item in sublist]

        #####################
        #Model centralized
    base_line = AMRulesClassifier(grace_period=200,
                            tie_threshold=0.05,
                            expand_confidence=0.01,
                            ordered_rules=ORDERED_RULES,
                            drift_detector=None,
                            nominal_attributes=[1,4,5,6,7,8,12], # adults
                            #nominal_attributes=[2,3,4,7,8,10,12,15,16,17], # heart
                            #nominal_attributes=[], # wine
                            #nominal_attributes=[0,3,4,5,6,7,10], # stroke
                            nb_prediction=False,
                            nb_threshold=0,
                            max_rules=50
                            )
    base_line.partial_fit(X, y) 
    X = x_test.values.tolist()
    y = y_test.values.tolist() 

    base_line_pred = base_line.predict(X) #blue
    baseline_pre = round(precision_score(y, base_line_pred, average=AVERAGE),6)
    baseline_rec = round(recall_score(y, base_line_pred, average=AVERAGE),6)
    baseline_f1 = round(f1_score(y, base_line_pred, average=AVERAGE),6)
    baseline_auc = round(roc_auc_score(y, base_line_pred, average=AVERAGE),6)
    baseline_nrules = len(base_line.rule_set)
        
    for n_models in NUMBER_MODELS:
        X = x_train.values.tolist()
        y = y_train.values.tolist()
        y = [item for sublist in y for item in sublist]

        #####################
        # Models
        LOGGER.info("\n\ntraining "+str(n_models)+" models")

        learners = list()
        for i in range(n_models):
            learners += [
                AMRulesClassifier(grace_period=200,
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=ORDERED_RULES,
                                drift_detector=None,
                                nominal_attributes=[1,4,5,6,7,8,12], # adults
                                #nominal_attributes=[2,3,4,7,8,10,12,15,16,17], # heart
                                #nominal_attributes=[], # wine
                                #nominal_attributes=[0,3,4,5,6,7,10], # stroke
                                nb_prediction=False,
                                nb_threshold=0,
                                max_rules=50
                                )
            ]

        #####################
        # Train
        l = int(len(X)/n_models)
        model_idx = 0 # starting learner who has biggest probability of receiving a instance
        start = time.time()

        if NODE_PROBABILITY != 1:
            for i in range(len(X)):
                X_i = [X[i]]
                y_i = [y[i]]

                if uniform(0, 1) <= NODE_PROBABILITY:
                    # change learner who has biggest probability of receiving the instance
                    x_idx = (model_idx+1)*l
                    if i == x_idx:
                        model_idx += 1
                        # there are no more learners
                        if model_idx >= n_models:
                            break # stops only this for-loop to train the learners
                    # train    
                    learners[model_idx].partial_fit(X_i, y_i)
                else:
                    # train
                    # any model but the one with highest prob
                    allowed_values = list(range(0, n_models))
                    allowed_values.remove(model_idx)
                    random_model_idx = choice(allowed_values)  

                    learners[random_model_idx].partial_fit(X_i, y_i)
        else:
            for i in range(len(X)):
                X_i = [X[i]]
                y_i = [y[i]]

                x_idx = (model_idx+1)*l
                if i == x_idx:
                    model_idx += 1
                    # there are no more learners
                    if model_idx >= n_models:
                        break # stops only this for-loop to train the learners
                # train 
                learner = learners[model_idx]
                learner.partial_fit(X_i, y_i)

        end = time.time()

        

        #####################
        # Merge models

        # Baseline
        baseline_merged_learner = AMRulesClassifier(grace_period=200,
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=ORDERED_RULES,
                                drift_detector=None,
                                nominal_attributes=[1,4,5,6,7,8,12], # adults
                                #nominal_attributes=[2,3,4,7,8,10,12,15,16,17], # heart
                                #nominal_attributes=[], # wine
                                #nominal_attributes=[0,3,4,5,6,7,10], # stroke
                                nb_prediction=False,
                                nb_threshold=0,
                                max_rules=50
                                )

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

    
        
        # Best default rule and no rule repetition
        best_df_no_rep_merged_learner = AMRulesClassifier(grace_period=200,
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=ORDERED_RULES,
                                drift_detector=None,
                                nominal_attributes=[1,4,5,6,7,8,12], # adults
                                #nominal_attributes=[2,3,4,7,8,10,12,15,16,17], # heart
                                #nominal_attributes=[], # wine
                                #nominal_attributes=[0,3,4,5,6,7,10], # stroke
                                nb_prediction=False,
                                nb_threshold=0,
                                max_rules=50
                                )
        

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
            #print(best_df_no_rep_merged_learner.default_rule.observed_class_distribution)
            merged_weight = next(iter(best_df_no_rep_merged_learner.default_rule.observed_class_distribution.values()))
            weight = next(iter(learner.default_rule.observed_class_distribution.values()))
            #print(merged_weight)
            #print(weight)
            if merged_weight < weight:
                best_df_no_rep_merged_learner.default_rule = copy.deepcopy(learner.default_rule)

        
        #####################
        # Predict
        acc_list = []
        pre_list = []
        rec_list = []
        f1_list = []
        auc_list = []
        n_rules_list = []

        X = x_test.values.tolist()
        y = y_test.values.tolist()

        # Centralized
        LOGGER.info("\nbaseline"+
                    "; \nprecision="+str(baseline_pre)+
                    "; \nrecall="+str(baseline_rec)+
                    "; \nf1_score="+str(baseline_f1)+
                    "; \nroc_auc="+str('baseline_auc'))
        #acc_list.append(acc)
        pre_list.append(baseline_pre)
        rec_list.append(baseline_rec)
        f1_list.append(baseline_f1)
        auc_list.append(baseline_auc)
        n_rules_list.append(len(base_line.rule_set))

        # Baseline merged, orange
        baseline_merged_learner_pred = baseline_merged_learner.predict(X)
        #acc = round(accuracy_score(y, baseline_merged_learner_pred),6)
        pre = round(precision_score(y, baseline_merged_learner_pred, average=AVERAGE),6)
        rec = round(recall_score(y, baseline_merged_learner_pred, average=AVERAGE),6)
        f1 = round(f1_score(y, baseline_merged_learner_pred, average=AVERAGE),6)
        auc = round(roc_auc_score(y, baseline_merged_learner_pred, average=AVERAGE),6)
        #auc = round(roc_auc_score(y, best_df_no_rep_merged_learner_pred, multi_class="ovr", average=AVERAGE),6) #wine
        LOGGER.info("\nbaseline_merged_learner"+
                    "; \nprecision="+str(pre)+
                    "; \nrecall="+str(rec)+
                    "; \nf1_score="+str(f1)+
                    "; \nroc_auc="+str('auc'))
        #acc_list.append(acc)
        pre_list.append(pre)
        rec_list.append(rec)
        f1_list.append(f1)
        auc_list.append(auc)
        n_rules_list.append(len(baseline_merged_learner.rule_set))

        # Best default rule and no rule repetition, red
        best_df_no_rep_merged_learner_pred = best_df_no_rep_merged_learner.predict(X)
        #acc = round(accuracy_score(y, best_df_no_rep_merged_learner_pred),6)
        pre = round(precision_score(y, best_df_no_rep_merged_learner_pred, average=AVERAGE),6)
        rec = round(recall_score(y, best_df_no_rep_merged_learner_pred, average=AVERAGE),6)
        f1 = round(f1_score(y, best_df_no_rep_merged_learner_pred, average=AVERAGE),6)
        auc = round(roc_auc_score(y, best_df_no_rep_merged_learner_pred, average=AVERAGE),6)
        #auc = round(roc_auc_score(y, best_df_no_rep_merged_learner_pred, multi_class="ovr", average=AVERAGE),6) #wine
        LOGGER.info("\nbest_df_no_rep_merged_learner"+
                    "; \nprecision="+str(pre)+
                    "; \nrecall="+str(rec)+
                    "; \nf1_score="+str(f1)+
                    "; \nroc_auc="+str('auc'))
        #acc_list.append(acc)
        pre_list.append(pre)
        rec_list.append(rec)
        f1_list.append(f1)
        auc_list.append(auc)
        n_rules_list.append(len(best_df_no_rep_merged_learner.rule_set))

        #list_test_acc.append(acc_list)
        list_test_pre.append(pre_list)
        list_test_rec.append(rec_list)
        list_test_f1.append(f1_list)
        list_test_auc.append(auc_list)
        list_n_rules.append(n_rules_list)

    
    
    LOGGER.info("; \nprecision="+str(list_test_pre)+
                    "; \nrecall="+str(list_test_rec)+
                    "; \nf1_score="+str(list_test_f1)+
                    "; \nroc_auc_score="+str(list_test_auc)+
                    "; \nnrules="+str(list_n_rules))

    k=len(NUMBER_MODELS)
    n = 3
    colors_list = np.array(["tab:blue", "tab:orange", "tab:red"]*k)
    #n_models_list = np.array([2]*n)
    n_models_list = np.array([2]*n + [4]*n + [6]*n)
    '''
    plt.subplot(1,4,1)
    acc_l = sum(list_test_acc, []) # np.array(list_test_mae).ravel()
    max_lim = int(max(acc_l) + max(acc_l)/3)
    plt.ylim([0,max_lim])
    plt.scatter(n_models_list, acc_l, c=colors_list)
    plt.xlabel('Number of nodes')
    plt.ylabel('Accuracy')
    plt.title('Testing Accuracy')
    plt.tight_layout()
    '''

    plt.subplot(1,4,1)
    pre_l = sum(list_test_pre, []) # np.array(list_test_mae).ravel()
    max_lim = int(max(pre_l) + max(pre_l)/3)
    plt.ylim([0,1])
    plt.scatter(n_models_list, pre_l, c=colors_list)
    plt.xlabel('Number of nodes')
    plt.ylabel('Precison')
    plt.title('Testing Precision')
    plt.tight_layout()

    plt.subplot(1,4,2)
    rec_l = sum(list_test_rec, []) # np.array(list_test_mae).ravel()
    max_lim = int(max(rec_l) + max(rec_l)/3)
    plt.ylim([0,1])
    plt.scatter(n_models_list, rec_l, c=colors_list)
    plt.xlabel('Number of nodes')
    plt.ylabel('Recall')
    plt.title('Testing Recall')
    plt.tight_layout()

    plt.subplot(1,4,3)
    f1_l = sum(list_test_f1, []) # np.array(list_test_mae).ravel()
    max_lim = int(max(f1_l) + max(f1_l)/3)
    plt.ylim([0,1])
    plt.scatter(n_models_list, f1_l, c=colors_list)
    plt.xlabel('Number of nodes')
    plt.ylabel('F1-Score')
    plt.title('Testing F1-Score')
    plt.tight_layout()

    
    plt.subplot(1,4,4)
    auc_l = sum(list_test_auc, []) # np.array(list_test_mae).ravel()
    max_lim = int(max(auc_l) + max(auc_l)/3)
    plt.ylim([0,1])
    plt.scatter(n_models_list, auc_l, c=colors_list)
    plt.xlabel('Number of nodes')
    plt.ylabel('Roc Auc')
    plt.title('Testing Roc Auc')
    plt.tight_layout()
    

    '''
    plt.subplot(1,4,4)
    rules_l = sum(list_n_rules, []) # np.array(list_n_rules).ravel()
    max_lim = int(max(rules_l) + max(rules_l)/3)
    plt.ylim([0,max_lim])
    plt.scatter(n_models_list, rules_l, c=colors_list)
    plt.xlabel('Number of nodes')
    plt.ylabel('Number of rules')
    plt.title('Number of rules across nodes')
    plt.tight_layout()
    '''
    plt.show()
test_merge_amrules() 