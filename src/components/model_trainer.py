import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import OneHotEncoder
from src.utils import save_object

from dataclasses import dataclass
import sys
import os


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self, train_array, test_array):
        try:
            logging.info(
                'Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Create an instance of the OneHotEncoder
            one_hot_encoder = OneHotEncoder()

            # Fit and transform the encoder on the categorical columns
            X_train_encoded = one_hot_encoder.fit_transform(X_train)
            X_test_encoded = one_hot_encoder.transform(X_test)

            # Define your model (e.g., Decision Tree)
            model = DecisionTreeClassifier()

            # Train the model
            model.fit(X_train_encoded, y_train)

            # Evaluate the model
            score = model.score(X_test_encoded, y_test)

            print(f'Model Score: {score}')

            # Save the trained model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

        except Exception as e:
            logging.info('Exception occurred at Model Training')
            raise CustomException(e, sys)
