import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import boto3
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, recall_score, f1_score, average_precision_score

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    recall = recall_score(actual, pred)
    f1 = f1_score(actual, pred)
    return accuracy, recall, f1


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the data_banknote_authentication.txt file from the URL
    colnames = ["variance", "skewness", "curtosis", "entropy", "class"]
    data_url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
    )

    try:
        data = pd.read_csv(data_url, sep=",", header=None, names=colnames)
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    train_x = train.drop(["class"], axis=1)
    test_x = test.drop(["class"], axis=1)
    train_y = train[["class"]]
    test_y = test[["class"]]

    print('Train shape:', train_x.shape)
    print('Train shape:', test_x.shape)

    n_estimators_argv = str(sys.argv[1]) if len(sys.argv) > 1 else '50,100'
    max_depth_argv = str(sys.argv[2]) if len(sys.argv) > 2 else '2,5'

    n_stimators_list = [int(x) for x in n_estimators_argv.split(',')]
    max_depth_list = [int(x) for x in max_depth_argv.split(',')]

    parameters = {'n_estimators': n_stimators_list, 'max_depth': max_depth_list}

    mlflow.set_experiment('randomForest_banknote_autolog')
    mlflow.sklearn.autolog()

    rf = RandomForestClassifier()
    rf_gridsearch = GridSearchCV(rf, parameters)

    with mlflow.start_run() as run:
        rf_gridsearch.fit(train_x, train_y)

