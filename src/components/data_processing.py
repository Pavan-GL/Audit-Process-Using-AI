import pandas as pd
import numpy as np
import logging
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

class FinancialDataProcessor:
    def __init__(self, data_path, output_dir):
        self.data_path = data_path
        self.output_dir = output_dir
        self.logger = self.setup_logger()

    def setup_logger(self):
        logger = logging.getLogger('FinancialDataProcessor')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(self.output_dir, 'data_processing.log'))
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def load_data(self):
        try:
            data = pd.read_csv(self.data_path)
            self.logger.info("Data loaded successfully.")
            return data
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def preprocess_data(self):
        try:
            data = self.load_data()
            self.logger.info("Starting data preprocessing.")

            # Basic preprocessing: handling missing values and scaling numerical data
            data.fillna(0, inplace=True)
            self.logger.info("Missing values filled.")

            scaler = StandardScaler()
            numerical_cols = ['amount', 'balance', 'transaction_frequency']
            data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
            self.logger.info("Numerical features scaled.")

            # Preprocessing for text data (if present)
            if 'document_text' in data.columns:
                documents = data['document_text']
                tfidf = TfidfVectorizer(max_features=500)
                text_features = tfidf.fit_transform(documents).toarray()
                self.logger.info("Text features extracted using TF-IDF.")
            else:
                text_features = np.array([])  # Placeholder for text features if column is missing

            # Add processed text features to the main dataset
            if text_features.size > 0:
                processed_data = pd.concat([data, pd.DataFrame(text_features)], axis=1)
            else:
                processed_data = data

            # Splitting data into training and testing sets
            X = processed_data.drop('target', axis=1)
            y = processed_data['target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.logger.info("Data split into training and testing sets.")

            # Save the processed data as a pickle file
            output_file = os.path.join(self.output_dir, 'processed_data.pkl')
            with open(output_file, 'wb') as f:
                pickle.dump((X_train, X_test, y_train, y_test), f)
            self.logger.info(f"Processed data saved to {output_file}.")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            self.logger.error(f"Error during preprocessing: {e}")
            raise

# Example usage
"""if __name__ == "__main__":
    processor = FinancialDataProcessor(data_path=r"D:\Audit Process\data\financial_data.csv", output_dir=r"D:\Audit Process\data")
    X_train, X_test, y_train, y_test = processor.preprocess_data()"""
