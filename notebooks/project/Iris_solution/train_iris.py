import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from urllib.parse import urlparse
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
    recall = recall_score(actual, pred, average='weighted')
    f1 = f1_score(actual, pred, average='weighted')
    return accuracy, recall, f1


if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    np.random.seed(40)

    colnames = ["sepal_length_in_cm", "sepal_width_in_cm", "petal_length_in_cm", "petal_width_in_cm", "class"]
    data_url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    )

    try:
        data = pd.read_csv(data_url, sep=",", header=None, names=colnames)
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    data = data.replace({"class": {"Iris-setosa": 1, "Iris-versicolor": 2, "Iris-virginica": 3}})

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    train_x = train.drop(["class"], axis=1)
    test_x = test.drop(["class"], axis=1)
    train_y = train[["class"]]
    test_y = test[["class"]]

    print('Train shape:', train_x.shape)
    print('Train shape:', test_x.shape)
    
    print(mlflow.get_tracking_uri())

    kernel = str(sys.argv[1]) if len(sys.argv) > 1 else 'linear'
    C = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        mlflow.set_experiment('iris_svm_project')
        svm = SVC(C=C, kernel=kernel, random_state=42)
        svm.fit(train_x, train_y)

        predicted_y = svm.predict(test_x)

        (accuracy, recall, f1) = eval_metrics(test_y, predicted_y)

        print("SVM model (Kernel=%s, C=%f):" % (kernel, C))
        print("  accuracy: %s" % accuracy)
        print("  recall: %s" % recall)
        print("  f1: %s" % f1)

        mlflow.log_param("kernel", kernel)
        mlflow.log_param("C", C)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(svm, "model", registered_model_name="iris_svm_model")
        else:
            mlflow.sklearn.log_model(svm, "model")
