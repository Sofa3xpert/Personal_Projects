from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Instantiate client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# List fine-tuned models
for job in client.fine_tuning.jobs.list():
    model = job.fine_tuned_model
    # Ignore failed jobs
    if model is not None:
         print(model)

# Line break
print("")

# List all models we have access to for inference
for model in client.models.list():
    print(model)
