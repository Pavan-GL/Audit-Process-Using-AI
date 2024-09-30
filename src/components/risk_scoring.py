import pandas as pd
import numpy as np
import logging
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, accuracy_score

class RiskScoringModel:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.logger = self.setup_logger()
        self.pipeline = None

    def setup_logger(self):
        logger = logging.getLogger('RiskScoringModel')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(self.output_dir, 'risk_scoring.log'))
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def train_model(self, X_train, y_train):
        try:
            # Ensure all column names are strings
            X_train.columns = X_train.columns.astype(str)
            
            # Identify categorical and numerical features
            categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
            numerical_features = X_train.select_dtypes(exclude=['object']).columns.tolist()

            # Create a ColumnTransformer for preprocessing
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', 'passthrough', numerical_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # Handle unknown categories
                ]
            )

            # Create a pipeline that first transforms data then fits the model
            self.pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', LogisticRegression())
            ])

            # Fit the model
            self.pipeline.fit(X_train, y_train)
            self.logger.info("Logistic Regression model trained successfully.")
        except Exception as e:
            self.logger.error(f"Error during model training: {e}")
            raise


    def predict_risk_scores(self, X_test):
        try:
            # Ensure all column names are strings
            X_test.columns = X_test.columns.astype(str)

            # Predict using the pipeline
            y_pred_proba = self.pipeline.predict_proba(X_test)[:, 1]
            self.logger.info("Risk scores predicted successfully.")
            return y_pred_proba
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            raise




    def evaluate_model(self, y_true, y_pred_proba):
        try:
            print(f"y_test distribution: {pd.Series(y_true).value_counts()}")
            roc_auc = roc_auc_score(y_true, y_pred_proba)
            print(f'ROC AUC Score: {roc_auc}')
        except ValueError as e:
            print(f"Error during ROC AUC evaluation: {e}")
            # Calculate accuracy as a fallback
            accuracy = accuracy_score(y_true, (y_pred_proba > 0.5).astype(int))
            print(f'Accuracy: {accuracy}')



    def categorize_risk(self, y_pred_proba):
        try:
            risk_thresholds = [0.33, 0.66]
            risk_scores = pd.cut(y_pred_proba, bins=[0, risk_thresholds[0], risk_thresholds[1], 1],
                                  labels=["low", "medium", "high"])
            self.logger.info("Risk scores categorized successfully.")
            return risk_scores
        except Exception as e:
            self.logger.error(f"Error during risk categorization: {e}")
            raise

    def save_model(self):
        try:
            model_file = os.path.join(self.output_dir, 'logistic_regression_model.pkl')
            with open(model_file, 'wb') as f:
                pickle.dump(self.pipeline, f)
            self.logger.info(f"Model saved to {model_file}.")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise

if __name__ == "__main__":
    try:
        with open(r"D:\Audit Process\data\processed_data.pkl", 'rb') as f:
            X_train, X_test, y_train, y_test = pickle.load(f)
    except Exception as e:
        print(f"Error loading processed data: {e}")

    output_dir = r"D:\Audit Process\data"
    risk_model = RiskScoringModel(output_dir=output_dir)

    risk_model.train_model(X_train, y_train)
    y_pred_proba = risk_model.predict_risk_scores(X_test)
    risk_model.evaluate_model(y_test, y_pred_proba)

    risk_categories = risk_model.categorize_risk(y_pred_proba)
    print(risk_categories.value_counts())

    risk_model.save_model()
