from skmultiflow.rules import AMRulesRegressor
from skmultiflow.drift_detection import PageHinkley
from skmultiflow.evaluation import EvaluatePrequential
from pandas_streaming.df import StreamingDataFrame
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from array import array

import time

import logging
LOGGER = logging.getLogger(__name__)


# Data


def test_amrules_drift():

    # Model
    learner_no_drift = AMRulesRegressor(grace_period=200,
                                tie_threshold=0.05,
                                expand_confidence=0.0000001,
                                ordered_rules=True,
                                drift_detector=None,
                                nominal_attributes=None,
                                decimal_numbers=4)

    
    learner_drift = AMRulesRegressor(grace_period=200,
                                tie_threshold=0.05,
                                expand_confidence=0.0000001,
                                ordered_rules=True,
                                drift_detector=PageHinkley(),
                                nominal_attributes=None,
                                decimal_numbers=4)

    # Set the evaluator
    evaluator = EvaluatePrequential(n_wait=200,
                                    max_time=3600,
                                    show_plot=True,
                                    metrics=['mean_square_error', 'mean_absolute_error'])

    
    # sdf = StreamingDataFrame.read_csv("/Users/mariajoaolavoura/Desktop/datasets/...", sep="\t", encoding="utf-8")
    # for df in sdf:

    # chunksize = 20000
    # for chunk in pd.read_csv('/Users/mariajoaolavoura/Desktop/datasets/', 
    #                         chunksize=chunksize, 
    #                         iterator=True):

    df = pd.read_csv("/Users/mariajoaolavoura/Desktop/datasets/dataverse_files_airline/2008.csv.bz2")    

    LOGGER.info("\ndf=\n"+str(df.head()))

    stream = df.values.tolist()

    # Run evaluation
    start = time.time()
    evaluator.evaluate(stream=stream, 
                        model=[learner_no_drift, learner_drift], 
                        model_names=['no_drift', 'drift'])
    end = time.time()
    LOGGER.info("\ntime: %.5f" % ((end - start)))


