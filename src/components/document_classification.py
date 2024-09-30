import logging
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentClassifier:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.model = self.create_pipeline()

    def create_pipeline(self):
        logging.info("Creating the text classification pipeline.")
        return Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000)),
            ('clf', MultinomialNB())
        ])

    def train_model(self):
        try:
            logging.info("Training the model...")
            self.model.fit(self.X_train, self.y_train)
            logging.info("Model training completed.")
        except Exception as e:
            logging.error(f"Error during model training: {e}")

    def predict(self, documents):
        try:
            logging.info("Making predictions...")
            predicted_categories = self.model.predict(documents)
            return predicted_categories
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return None

    def evaluate_model(self, y_true, y_pred):
        logging.info("Evaluating the model...")
        report = classification_report(y_true, y_pred)
        logging.info("\n" + report)

    def save_model(self, filename='document_classifier.pkl'):
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.model, f)
            logging.info(f"Model saved to {filename}.")
        except Exception as e:
            logging.error(f"Error saving model: {e}")

if __name__ == "__main__":
    # Sample data loading (replace this with actual data)
    data = pd.DataFrame({
        'document_text': ['Example text 1', 'Example text 2'],
        'category': ['Compliant', 'Non-Compliant']
    })
    
    # Train-test split (replace this with your actual split logic)
    X_train = data['document_text']
    y_train = data['category']

    classifier = DocumentClassifier(X_train, y_train)
    classifier.train_model()

    # Test predictions (replace this with your actual test data)
    X_test = pd.Series(['Test document text'])
    predicted_categories = classifier.predict(X_test)

    # Evaluate model (replace this with your actual test labels)
    y_test = pd.Series(['Compliant'])
    classifier.evaluate_model(y_test, predicted_categories)

    # Save the model
    classifier.save_model()
