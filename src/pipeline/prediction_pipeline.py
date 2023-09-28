import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        self.model_path = os.path.join('artifacts', 'model.pkl')
        self.preprocessor = load_object(self.preprocessor_path)
        self.model = load_object(self.model_path)

    def predict(self, features):
        try:
            # Assuming features is a DataFrame with the same columns as the training data
            data_scaled = self.preprocessor.transform(features)
            pred = self.model.predict(data_scaled)
            return pred

        except Exception as e:
            logging.info("Exception occurred in prediction")
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, interest, value, skill, personality):
        self.data_dict = {
            'interest': [interest],
            'value': [value],
            'skill': [skill],
            'personality': [personality],
        }

    def get_data_as_dataframe(self):
        try:
            df = pd.DataFrame(self.data_dict)
            logging.info('Dataframe Gathered')
            return df

        except Exception as e:
            logging.info('Exception Occurred in prediction pipeline')
            raise CustomException(e, sys)
