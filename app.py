from flask import Flask, render_template, jsonify, request
import openai
import logging
import os

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up OpenAI API Key
openai.api_key = os.getenv('OPENAI_API_KEY')
logger.info("OpenAI API key has been set.")

def query_audit_compliance(question):
    try:
        logger.info("Generating response for the compliance question...")
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": f"Answer the following compliance-related question for an auditor: {question}"}],
            max_tokens=100
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        logger.error(f"Error during API call: {e}")
        return None

# Endpoint for real-time anomaly detection results
@app.route('/anomalies', methods=['GET'])
def get_anomalies():
    anomalies = {"anomaly_count": 10, "high_risk_transactions": ["TXN1234", "TXN5678"]}
    return jsonify(anomalies)

# Dashboard for showing real-time data and asking questions
@app.route('/', methods=['GET', 'POST'])
def dashboard():
    answer = None
    if request.method == 'POST':
        question = request.form['question']
        answer = query_audit_compliance(question)
    return render_template('dashboard.html', answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
