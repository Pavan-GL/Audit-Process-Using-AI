# Audit Compliance Dashboard

## Overview
The Audit Compliance Dashboard is a web application designed to facilitate real-time anomaly detection and provide insights into audit compliance issues. It leverages natural language processing (NLP) capabilities through OpenAI's API to answer compliance-related questions and summarize audit findings.

The outcome of this AI-powered audit process project includes the following deliverables and insights:

1. Anomaly Detection in Financial Transactions

Outcome: The model identifies anomalous transactions using Isolation Forest, flagging potentially fraudulent or risky transactions.
Business Value: Detects unusual patterns in financial data to help auditors focus on suspicious transactions, enhancing the accuracy and efficiency of financial audits.
2. Risk Scoring for Transactions

Outcome: A risk scoring system for financial transactions, predicting the probability of each transaction being risky based on features like amount, balance, transaction frequency, and type.
Business Value: Prioritizes high-risk transactions for auditors, streamlining the audit process and improving risk management.
3. NLP-Based Document Classification

Outcome: Automatic classification of financial documents (e.g., transaction descriptions) into predefined categories such as "approved", "unauthorized", etc., using machine learning models trained on textual data.
Business Value: Automates the review of large volumes of audit-related documents, reducing manual effort and improving consistency in document classification.
4. Audit Report Generation with Generative AI

Outcome: An audit report generated based on the findings (anomalies, risk scores, document classifications) using a large language model (LLM) such as GPT. This report summarizes key insights and highlights suspicious activities or potential risks.
Business Value: Automates report generation, providing auditors with comprehensive summaries and insights, reducing time spent on writing detailed reports.
5. Q&A System for Compliance and Regulations

Outcome: An interactive LLM-powered Q&A system that answers questions related to compliance and regulations in the context of the audit process.
Business Value: Assists auditors by providing quick and accurate answers to regulatory and compliance-related queries, ensuring adherence to legal standards.
6. Audit Event Logging

Outcome: Logging system that tracks audit-related events, ensuring that the process is transparent, traceable, and well-documented.
Business Value: Facilitates accountability and traceability of audit activities, supporting compliance and governance efforts.

Overall Benefits and Business Value:
Automation of Routine Tasks: Automates tasks like anomaly detection, document classification, risk scoring, and report generation, saving significant time and reducing human errors.
Increased Audit Efficiency: Helps auditors focus on high-risk areas by flagging anomalies and scoring risky transactions, improving the overall efficiency and quality of the audit process.
Data-Driven Insights: Provides data-driven insights through machine learning and AI, leading to more informed decision-making and improved risk management.
Compliance and Regulation Support: The LLM-powered Q&A feature assists in real-time compliance support, ensuring that the audit process aligns with industry regulations.
This project significantly streamlines and enhances the audit process, especially in large-scale financial audits.

## Features
- **Real-Time Anomaly Detection**: Displays anomalies and high-risk transactions detected in financial data.
- **NLP-Based Q&A**: Users can ask compliance-related questions, and the application provides answers using the OpenAI API.
- **Interactive Dashboard**: An easy-to-use web interface that allows auditors to navigate through the data and insights effectively.

## Technologies Used
- **Flask**: A lightweight web framework for building the web application.
- **OpenAI API**: Utilized for natural language processing tasks such as summarizing findings and answering compliance questions.
- **Pandas**: Used for data manipulation and analysis.
- **HTML/CSS/JavaScript**: For building the front-end user interface.
- **Jinja2**: Templating engine used with Flask for rendering HTML pages.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/audit-compliance-dashboard.git
   cd audit-compliance-dashboard

Set up a virtual environment:
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate


Install the required packages:
pip install -r requirements.txt

Set your OpenAI API key:
export OPENAI_API_KEY='YOUR_API_KEY'  # On Windows use: set OPENAI_API_KEY='YOUR_API_KEY'

Input
User Queries: Users can input compliance-related questions through the dashboard interface.
Audit Findings: The system can summarize input audit findings using the OpenAI API.

Expected Outcomes

Anomaly Detection Results: Users can view real-time data on high-risk transactions and anomaly counts.
NLP Answers: Users receive comprehensive answers to their compliance questions, aiding in the audit process.
Summarized Reports: The application generates concise summaries of audit findings for easy review.

Usage
Start the application:

python app.py
Open your web browser and navigate to http://127.0.0.1:5000/.

Use the input form to ask compliance-related questions and view real-time anomaly data.

Logs and Exception Handling
The application is designed to log significant events and errors during execution for easier debugging and monitoring. Logs are generated in a file and printed to the console.

Contribution
Contributions are welcome! Feel free to open issues or submit pull requests.


Acknowledgments
OpenAI for providing powerful NLP tools.
Flask for the web framework.

The open-source community for various tools and libraries that made this project possible.