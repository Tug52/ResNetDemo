import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def mlops_log(num_epochs, val_acc, val_loss, train_acc, train_loss):
    # Create a new MLflow Experiment
    mlflow.set_experiment("MLflow Quickstart")

    # Start an MLflow run
    with mlflow.start_run():

        # Log the loss metric
        mlflow.log_metric("Validation Accuracy", val_acc)
        mlflow.log_metric("Validation Loss", val_loss)
        mlflow.log_metric("Training Accuracy", train_acc)
        mlflow.log_metric("Training Loss", train_loss)
        mlflow.log_metric("Number of Epochs", num_epochs)

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", "Basic model for resNetDemo")
