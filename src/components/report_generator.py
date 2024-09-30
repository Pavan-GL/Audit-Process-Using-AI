import logging
import openai
import pickle
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AuditReportGenerator:
    def __init__(self, api_key):
        openai.api_key = api_key
        logging.info("OpenAI API key has been set.")

    def summarize_findings(self, audit_findings):
        logging.info("Generating summary for audit findings...")
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",  # Specify the model you want to use
                messages=[
                    {"role": "user", "content": f"Summarize the following audit findings into a concise audit report:\n{audit_findings}"}
                ],
                max_tokens=150
            )
            report_summary = response.choices[0].message['content'].strip()
            logging.info("Summary generated successfully.")
            return report_summary
        except Exception as e:
            logging.error(f"Error during API call: {e}")
            return None

    def save_summary(self, summary, filename='audit_report_summary.pkl'):
        try:
            with open(filename, 'wb') as f:
                pickle.dump(summary, f)
            logging.info(f"Summary saved to {filename}.")
        except Exception as e:
            logging.error(f"Error saving summary: {e}")

if __name__ == "__main__":
    # Set your OpenAI API Key here
    API_KEY = os.getenv('OPENAI_API_KEY')
    
    # Example audit findings
    audit_findings = "The company had irregular transactions in Q3, resulting in a financial discrepancy of $100,000..."

    # Create an instance of the report generator
    report_generator = AuditReportGenerator(api_key=API_KEY)

    # Generate summary
    summary = report_generator.summarize_findings(audit_findings)
    
    if summary:
        print("Generated Audit Report Summary:\n", summary)
        # Optionally save the summary
        report_generator.save_summary(summary)
