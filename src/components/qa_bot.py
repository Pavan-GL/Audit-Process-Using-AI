import openai
import logging
import pickle
import os

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set up OpenAI API Key
openai.api_key = os.getenv('OPENAI_API_KEY')
logging.info("OpenAI API key has been set.")

def query_audit_compliance(question):
    try:
        logging.info(f"Querying compliance question: {question}")
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": f"Answer the following compliance-related question for an auditor: {question}"}
            ]
        )
        answer = response.choices[0].message['content'].strip()
        logging.info("Received compliance answer.")
        return answer
    except Exception as e:
        logging.error("Error during API call: " + str(e))
        return None

def save_answer_to_pickle(answer, filename):
    if answer:
        try:
            with open(filename, 'wb') as f:
                pickle.dump(answer, f)
            logging.info(f"Answer saved to {filename}.")
        except Exception as e:
            logging.error("Error saving answer to pickle: " + str(e))

# Example query
if __name__ == "__main__":
    question = input("Please enter your compliance-related question: ")
    answer = query_audit_compliance(question)
    
    if answer:
        print("Compliance Answer: ", answer)

        # Save the answer to a pickle file
        with open('compliance_answer.pkl', 'wb') as f:
            pickle.dump(answer, f)
        logging.info("Answer saved to compliance_answer.pkl.")
