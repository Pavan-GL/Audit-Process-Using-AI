import pandas as pd
import numpy as np
import logging
import os
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

class AnomalyDetector:
    def __init__(self, output_dir, contamination=0.01):
        self.output_dir = output_dir
        self.contamination = contamination
        self.logger = self.setup_logger()
        self.model = None

    def setup_logger(self):
        logger = logging.getLogger('AnomalyDetector')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(self.output_dir, 'anomaly_detection.log'))
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def train_model(self, X_train):
        try:
            # Ensure all column names are strings
            X_train.columns = X_train.columns.astype(str)

            # Drop non-numeric columns
            X_train_numeric = X_train.select_dtypes(include=[np.number])
            if X_train_numeric.empty:
                self.logger.error("No numeric features available for training.")
                raise ValueError("No numeric features available for training.")

            self.model = IsolationForest(contamination=self.contamination)
            self.model.fit(X_train_numeric)
            self.logger.info("Isolation Forest model trained successfully.")
        except Exception as e:
            self.logger.error(f"Error during model training: {e}")
            raise

    def predict_anomalies(self, X_test):
        try:
            # Ensure all column names are strings
            X_test.columns = X_test.columns.astype(str)

            # Drop non-numeric columns
            X_test_numeric = X_test.select_dtypes(include=[np.number])
            if X_test_numeric.empty:
                self.logger.error("No numeric features available for prediction.")
                raise ValueError("No numeric features available for prediction.")

            y_pred = self.model.predict(X_test_numeric)
            y_pred = np.where(y_pred == -1, 1, 0)  # Convert -1 to 1 (anomaly) and 1 to 0 (normal)
            self.logger.info("Anomaly predictions made successfully.")
            return y_pred
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            raise

    def evaluate_model(self, y_test, y_pred):
        try:
            report = classification_report(y_test, y_pred, zero_division=0)
            self.logger.info("Model evaluation completed.")
            print(report)
        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            raise


    def save_model(self):
        try:
            model_file = os.path.join(self.output_dir, 'isolation_forest_model.pkl')
            with open(model_file, 'wb') as f:
                pickle.dump(self.model, f)
            self.logger.info(f"Model saved to {model_file}.")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise

# Example usage
"""if __name__ == "__main__":
    # Load X_train, X_test, y_train, y_test from the pickle file created earlier
    try:
        with open(r"D:\Audit Process\data\processed_data.pkl", 'rb') as f:
            X_train, X_test, y_train, y_test = pickle.load(f)
    except Exception as e:
        print(f"Error loading processed data: {e}")

    # Initialize and run anomaly detection
    output_dir = r"D:\Audit Process\data"
    detector = AnomalyDetector(output_dir=output_dir)

    detector.train_model(X_train)
    y_pred = detector.predict_anomalies(X_test)
    detector.evaluate_model(y_test, y_pred)
    detector.save_model()"""
