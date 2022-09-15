from skmultiflow.rules import AMRulesClassifier, VeryFastDecisionRulesClassifier
from skmultiflow.drift_detection import PageHinkley, DDM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from math import sqrt
import numpy as np
import pandas as pd
from array import array

import time

import logging
logging.basicConfig(filename='logsinfo.log', filemode='w', level=logging.INFO)
LOGGER = logging.getLogger(__name__)

df = pd.read_csv("/home/angelo/Desktop/databases/adults.csv")
#df = pd.read_csv("/home/angelo/Desktop/databases/heart.csv")
#df = pd.read_csv("/home/angelo/Desktop/databases/stoke.csv")
#df = pd.read_csv("/home/angelo/Desktop/databases/wine.csv") 
'''
df = pd.read_csv("/Users/mariajoaolavoura/Desktop/datasets/fraud/dataset.csv", sep=",")
df = df.iloc[:15000,:]
df["device_id"] = df["device_id"].astype('category')
df["device_id"] = df["device_id"].cat.codes
df["source"] = df["source"].astype('category')
df["source"] = df["source"].cat.codes
df["browser"] = df["browser"].astype('category')
df["browser"] = df["browser"].cat.codes
df["sex"] = df["sex"].astype('category')
df["sex"] = df["sex"].cat.codes
'''
# """
def test_amrules_drift():

    LOGGER.info("\ndf=\n"+str(df.head()))

    x = df.iloc[:,:-1]
    y = df.iloc[:,-1:]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    X = x_train.values.tolist()
    y = y_train.values.tolist()
    y = [item for sublist in y for item in sublist]
    
    # Model
    learner_no_drift = AMRulesClassifier(grace_period=int(len(df)*0.02), 
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=True,
                                drift_detector=None,
                                # nominal_attributes=[1,4,5,6,7,8,12], # adults
                                # nominal_attributes=[2,3,4,7,8,10,12,15,16,17], # heart
                                 nominal_attributes=[], # wine
                                # nominal_attributes=[0,3,4,5,6,7,10], # stroke
                                nb_prediction=False,
                                nb_threshold=0)
    LOGGER.info("\nno drift\nmodel=\n"+str(learner_no_drift.get_info()))

    learner_drift = AMRulesClassifier(grace_period=int(len(df)*0.02), 
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=True,
                                drift_detector=PageHinkley(),
                                # nominal_attributes=[1,4,5,6,7,8,12], # adults
                                # nominal_attributes=[2,3,4,7,8,10,12,15,16,17], # heart
                                 nominal_attributes=[], # wine
                                # nominal_attributes=[0,3,4,5,6,7,10], # stroke
                                nb_prediction=False,
                                nb_threshold=0)
    LOGGER.info("\ndrift\nmodel=\n"+str(learner_drift.get_info()))

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
    y = [item for sublist in y for item in sublist]

    pred_no_drift = learner_no_drift.predict(X)
    LOGGER.info("\nno drift\nreport=\n"+str(classification_report(y, pred_no_drift)))

    pred_drift = learner_drift.predict(X)
    LOGGER.info("\ndrift\nreport=\n"+str(classification_report(y, pred_drift)))


"""
def test_vfdr_drift():

    LOGGER.info("\ndf=\n"+str(df.head()))

    x = df.iloc[:,:-1]
    y = df.iloc[:,-1:]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    X = x_train.values.tolist()
    y = y_train.values.tolist()
    y = [item for sublist in y for item in sublist]
    
    # Model
    learner_no_drift = VeryFastDecisionRulesClassifier(grace_period=int(len(df)*0.02), 
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=True,
                                drift_detector=None,
                                # nominal_attributes=[0,4,5], # ecoli
                                # nominal_attributes=[0], # abalone
                                # nominal_attributes=[], # iris
                                nominal_attributes=[1,2,3,4,6], # fraud
                                nb_prediction=False,
                                nb_threshold=0)

    
    learner_drift = VeryFastDecisionRulesClassifier(grace_period=int(len(df)*0.02),
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=True,
                                drift_detector=DDM(),
                                # nominal_attributes=[0,4,5], # ecoli
                                # nominal_attributes=[0], # abalone
                                # nominal_attributes=[], # iris
                                nominal_attributes=[1,2,3,4,6], # fraud
                                nb_prediction=False,
                                nb_threshold=0)

    LOGGER.info("\nno drift\nmodel=\n"+str(learner_no_drift.get_info()))
    LOGGER.info("\ndrift\nmodel=\n"+str(learner_drift.get_info()))

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

    # Predict
    X = x_test.values.tolist()
    y = y_test.values.tolist()
    y = [item for sublist in y for item in sublist]

    LOGGER.info("\ny test std=\n"+str(np.std(y)))
    
    pred_no_drift = learner_no_drift.predict(X)
    LOGGER.info("\nno drift\nreport=\n"+str(classification_report(y, pred_no_drift)))

    pred_drift = learner_drift.predict(X)
    LOGGER.info("\ndrift\nreport=\n"+str(classification_report(y, pred_drift)))
"""


def test_amrules_ordered_rules():

    LOGGER.info("\ndf=\n"+str(df.head()))

    x = df.iloc[:,:-1]
    y = df.iloc[:,-1:]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    X = x_train.values.tolist()
    y = y_train.values.tolist()
    y = [item for sublist in y for item in sublist]
    
    # Model
    learner_no_order = AMRulesClassifier(grace_period=int(len(df)*0.02),
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=False,
                                drift_detector=None,
                                nominal_attributes=[1,4,5,6,7,8,12], # adults
                                # nominal_attributes=[2,3,4,7,8,10,12,15,16,17], # heart
                                # nominal_attributes=[], # wine
                                # nominal_attributes=[0,3,4,5,6,7,10], # stroke
                                nb_prediction=False,
                                nb_threshold=0)

    
    learner_order = AMRulesClassifier(grace_period=int(len(df)*0.02),
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=True,
                                drift_detector=None,
                                nominal_attributes=[1,4,5,6,7,8,12], # adults
                                # nominal_attributes=[2,3,4,7,8,10,12,15,16,17], # heart
                                # nominal_attributes=[], # wine
                                # nominal_attributes=[0,3,4,5,6,7,10], # stroke
                                nb_prediction=False,
                                nb_threshold=0)

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

    LOGGER.info("\nsame models=\n"+str(learner_no_order.get_model_description() == learner_order.get_model_description()))

    # Predict
    X = x_test.values.tolist()
    y = y_test.values.tolist()
    y = [item for sublist in y for item in sublist]

    pred_no_order = learner_no_order.predict(X)
    LOGGER.info("\nno order\nreport=\n"+str(classification_report(y, pred_no_order)))

    pred_order = learner_order.predict(X)
    LOGGER.info("\norder\nreport=\n"+str(classification_report(y, pred_order)))


"""
def test_amrules_nb():

    LOGGER.info("\ndf=\n"+str(df.head()))

    x = df.iloc[:,:-1]
    y = df.iloc[:,-1:]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    X = x_train.values.tolist()
    y = y_train.values.tolist()
    y = [item for sublist in y for item in sublist]
    
    # Model
    learner_no_nb = AMRulesClassifier(grace_period=int(len(df)*0.02),
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=True,
                                drift_detector=None,
                                # nominal_attributes=[0,4,5], # ecoli
                                # nominal_attributes=[0], # abalone
                                # nominal_attributes=[], # iris
                                nominal_attributes=[1,2,3,4,6], # fraud
                                nb_prediction=False,
                                nb_threshold=0)

    
    learner_nb = AMRulesClassifier(grace_period=int(len(df)*0.02),
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=True,
                                drift_detector=None,
                                # nominal_attributes=[0,4,5], # ecoli
                                # nominal_attributes=[0], # abalone
                                # nominal_attributes=[], # iris
                                nominal_attributes=[1,2,3,4,6], # fraud #[0,1,2,3,4,5,6,7,8,9]) \
                                nb_prediction=True,
                                nb_threshold=0.5)

    LOGGER.info("\nno nb\nmodel=\n"+str(learner_no_nb.get_info()))
    LOGGER.info("\nnb\nmodel=\n"+str(learner_nb.get_info()))

    # Train
    learner_no_nb.partial_fit(X, y)
    LOGGER.info("\nno nb\nmodel=\n"+str(learner_no_nb.get_model_description()))

    learner_no_nb.partial_fit(X, y)
    LOGGER.info("\nnb\nmodel=\n"+str(learner_nb.get_model_description()))

    # Predict
    X = x_test.values.tolist()
    y = y_test.values.tolist()
    y = [item for sublist in y for item in sublist]

    pred_no_nominal = learner_no_nb.predict(X)
    LOGGER.info("\nno nb\nMAE="+str(round(mean_absolute_error(y, pred_no_nominal),6)))
    LOGGER.info("\nno nb\nRMSE="+str(round(sqrt(mean_squared_error(y, pred_no_nominal)),6)))

    pred_nominal = learner_no_nb.predict(X)
    LOGGER.info("\nnb\nMAE="+str(round(mean_absolute_error(y, pred_nominal),6)))
    LOGGER.info("\nnb\nRMSE="+str(round(sqrt(mean_squared_error(y, pred_nominal)),6)))




def test_vfdr_nb():

    LOGGER.info("\ndf=\n"+str(df.head()))

    x = df.iloc[:,:-1]
    y = df.iloc[:,-1:]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    X = x_train.values.tolist()
    y = y_train.values.tolist()
    y = [item for sublist in y for item in sublist]
    
    # Model
    learner_no_nb = VeryFastDecisionRulesClassifier(grace_period=int(len(df)*0.02),
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=True,
                                drift_detector=None,
                                # nominal_attributes=[0,4,5], # ecoli
                                # nominal_attributes=[0], # abalone
                                # nominal_attributes=[], # iris
                                nominal_attributes=[1,2,3,4,6], # fraud
                                nb_prediction=False,
                                nb_threshold=0)

    
    learner_nb = VeryFastDecisionRulesClassifier(grace_period=int(len(df)*0.02),
                                tie_threshold=0.05,
                                expand_confidence=0.01,
                                ordered_rules=True,
                                drift_detector=None,
                                # nominal_attributes=[0,4,5], # ecoli
                                # nominal_attributes=[0], # abalone
                                # nominal_attributes=[], # iris
                                nominal_attributes=[1,2,3,4,6], # fraud #[0,1,2,3,4,5,6,7,8,9])
                                nb_prediction=True,
                                nb_threshold=0.98)

    LOGGER.info("\nno nb\nmodel=\n"+str(learner_no_nb.get_info()))
    LOGGER.info("\nnb\nmodel=\n"+str(learner_nb.get_info()))

    # Train
    learner_no_nb.partial_fit(X, y)
    LOGGER.info("\nno nb\nmodel=\n"+str(learner_no_nb.get_model_description()))

    learner_no_nb.partial_fit(X, y)
    LOGGER.info("\nnb\nmodel=\n"+str(learner_nb.get_model_description()))

    # Predict
    X = x_test.values.tolist()
    y = y_test.values.tolist()
    y = [item for sublist in y for item in sublist]

    pred_no_nominal = learner_no_nb.predict(X)
    LOGGER.info("\nno nominal\nMAE="+str(round(mean_absolute_error(y, pred_no_nominal),6)))
    LOGGER.info("\nno nominal\nRMSE="+str(round(sqrt(mean_squared_error(y, pred_no_nominal)),6)))

    pred_nominal = learner_no_nb.predict(X)
    LOGGER.info("\nnominal\nMAE="+str(round(mean_absolute_error(y, pred_nominal),6)))
    LOGGER.info("\nnominal\nRMSE="+str(round(sqrt(mean_squared_error(y, pred_nominal)),6)))


def test_amrules_one_simple_sample():
    X = [
        [155,10,3,5,9],
        [155,10,3,5,9],
        [180,9,4,10,9],
        [180,8,5,10,10],
        [190,7,6,5,11],
        [190,6,6,5,12],
        [220,5,4,10,13],
        [0,4,6,5,14],
        [225,3,3,10,15],
        [0,8,5,10,10],
        [225,3,3,10,15],
        [220,5,4,10,13],
    ]
    y = [1,1,0,0,0,0,0,1,0,0,0,0]

    learner = AMRulesClassifier(grace_period=2, drift_detector=None, nb_prediction=False)

    # Train
    learner.partial_fit(X, y)

    LOGGER.info(learner.get_model_description())



def test_amrules_one_sample():

    # Setup the stream
    stream = AGRAWALGenerator()
    X, y = stream.next_sample(5000)

    # Setup the learner
    learner = AMRulesClassifier(drift_detector=None,nb_prediction=False)

    # Train
    learner.partial_fit(X, y)

    # Print rules
    print(learner.get_model_description())
    LOGGER.info(learner.get_model_description())

    # Predict
    X, y = stream.next_sample(10)
    predicted = learner.predict(X)

    assert np.alltrue(predicted == y)


def test_amrules_max_5000():

    # Setup the learners
    learner = AMRulesClassifier(drift_detector=None,nb_prediction=False)

    # Setup the stream
    stream = AGRAWALGenerator(random_state=11)

    cnt = 0
    max_samples = 5000
    predictions = array('i')
    proba_predictions = []
    wait_samples = 100

    while cnt < max_samples:
        X, y = stream.next_sample()
        
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            predictions.append(learner.predict(X)[0])
            proba_predictions.append(learner.predict_proba(X)[0])

        # Train
        learner.partial_fit(X, y)
        cnt += 1

    # Print rules
    # LOGGER.info(learner.get_model_description())
    print(learner.get_model_description())

    expected_predictions = array('i', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
                                       0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0,
                                       0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0])

    assert np.alltrue(predictions == expected_predictions)


def test_amrules_info_gain():

    learner = AMRulesClassifier(drift_detector=None,
                                nominal_attributes=[3, 4, 5],
                                nb_prediction=False)

    stream = AGRAWALGenerator(random_state=11)

    cnt = 0
    max_samples = 5000
    predictions = array('i')
    proba_predictions = []
    wait_samples = 100

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            predictions.append(learner.predict(X)[0])
            proba_predictions.append(learner.predict_proba(X)[0])
        learner.partial_fit(X, y)
        cnt += 1

    expected_predictions = array('i', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
                                       0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0,
                                       0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0])

    assert np.alltrue(predictions == expected_predictions)


    expected_info = "AMRulesClassifier(drift_detector=None, expand_confidence=1e-07, " \
                    "expand_criterion='info_gain', grace_period=200, max_rules=1000, min_weight=100, " \
                    "nb_prediction=False, nb_threshold=0, nominal_attributes=[3, 4, 5], ordered_rules=True, " \
                    "remove_poor_atts=True, rule_prediction='first_hit', tie_threshold=0.05)"

    info = " ".join([line.strip() for line in learner.get_info().split()])

    LOGGER.info(learner.get_info())
    assert info == expected_info


    expected_model_description_1 = 'Rule 0 :Att (2) <= 39.550| class :0  {0: 1365.7101742993455}\n' + \
                                   'Rule 1 :Att (2) <= 58.180| class :1  {1: 1269.7307449971418}\n' + \
                                   'Rule 2 :Att (2) <= 60.910| class :0  {0: 66.24158839706533, 1: 54.0}\n' + \
                                   'Default Rule :| class :0  {0: 1316.7584116029348}'

    expected_model_description_2 = 'Rule 0 :Att (2) <= 39.550| class :0  {0: 1365.7101742993455}\n' + \
                                   'Rule 1 :Att (2) <= 58.180| class :1  {1: 1269.7307449971418}\n' + \
                                   'Rule 2 :Att (2) <= 60.910| class :0  {0: 66.241588397065328, 1: 54.0}\n' + \
                                   'Default Rule :| class :0  {0: 1316.7584116029348}'

    LOGGER.info(learner.get_model_description())
    assert (learner.get_model_description() == expected_model_description_1) or \
           (learner.get_model_description() == expected_model_description_2)

    # Following test only covers 'Number of rules' since 'model_size in bytes' is calculated using
    # the 'calculate_object_size' utility function which is validated in its own test
    expected_number_of_rules = 3
    assert learner.get_model_measurements()['Number of rules: '] == expected_number_of_rules

"""

test_amrules_ordered_rules()