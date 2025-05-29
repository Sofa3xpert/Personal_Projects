from openai import OpenAI
from time import sleep
from dotenv import load_dotenv

# Load environment variables - client looks for OPENAI_API_KEY on instantiation.
load_dotenv()

# Instantiate client
client = OpenAI()

# Select model. DO NOT CHANGE FROM "gpt-4o-mini-2024-07-18"
model="gpt-4o-mini-2024-07-18"

# Function to upload fine-tuning files to OpenAI. Returns the file ID
def upload_training_file(file_path):
    with open(file_path, "rb") as file:
        response = client.files.create(
            file=file,
            purpose="fine-tune"
        )
        return response.id


# Function to create a fine-tuning job. Returns the job ID
def create_fine_tuning_job(id1, id2, model_name):
    response = client.fine_tuning.jobs.create(
        training_file=id1,
        validation_file=id2,
        model=model_name
    )
    return response.id

# Function to monitor the fine-tuning process
def monitor_job(monitor_id):
    while True:
        job = client.fine_tuning.jobs.retrieve(monitor_id)
        print(f"Status: {job.status}")

        if job.status in ["succeeded", "failed"]:
            return job

        # List latest events
        events = client.fine_tuning.jobs.list_events(
            fine_tuning_job_id=monitor_id,
            limit=5
        )
        for event in events.data:
            print(f"Event: {event.message}")

        sleep(30)  # Check every 30 seconds


# Upload training and validation files and store their IDs
training_file_id = upload_training_file("../resources/train.jsonl")
validation_file_id = upload_training_file("../resources/val.jsonl")

# Create the fine-tuning job
job_id = create_fine_tuning_job(training_file_id, validation_file_id, model)

# Monitor the job until completion. Can also be monitored in the fine-tuning UI: https://platform.openai.com/finetune
fine_tuning_job = monitor_job(job_id)

# Log info on success/failure
if fine_tuning_job.status == "succeeded":
    fine_tuned_model = fine_tuning_job.fine_tuned_model
    print(f"Fine-tuned model ID: {fine_tuned_model}")
else:
    print("Fine-tuning failed.")
