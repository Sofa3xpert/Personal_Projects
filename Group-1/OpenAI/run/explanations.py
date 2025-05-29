from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

prediction = None
statement = None

# Instantiate client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Define a helper function to call the API
def get_explanation(patient_statement, predicted_diagnosis):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert mental health assistant. You are a supplementary tool to the doctor or medical professional evaluating the patient."
                "Your task is to provide a concise, evidence-based explanation of how a patient's statement supports a given diagnosis. "
                "Focus on the key aspects of the statement that indicate the diagnosis. Use the relevant medical terminology where appropriate, as your users will be trained medical professionals."
                "Do not use bullet points or other formatting. Just provide a clear and concise explanation, helpful to the doctor or medical professional evaluating the patient."
            )
        },
        {
            "role": "user",
            "content": (
                f"Patient statement: {patient_statement}\n"
                f"Predicted diagnosis: {predicted_diagnosis}\n"
                "Please provide a brief explanation of how the statement supports this diagnosis."
            )
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=256
        )
        explanation = response.choices[0].message.content.strip()
    except Exception as e:
        explanation = "API call failed."
    return explanation


explanation = get_explanation("i hate it and i cannot do this huh, why should i stay alive oh yeah sunsets and puppies, that is rightever since when did sunsets and puppies outweigh, what, being screamed at by your parents, being told you are a failure, crying alone being ignored by everyone and knowing that the people you trust the most are also suicidal haampx200bi literally would have done it already but i had nothign to do it with and because i wish this was something i did not have a choice in title", "Suicidal")
print(explanation)