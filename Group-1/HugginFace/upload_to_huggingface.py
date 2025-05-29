from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Login to Hugging Face Hub with token from .env
login(token=os.environ.get("HF_TOKEN"))

model_path = "models/BERT/10.03.2025-15.50-ML128E1"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")


repo_name = "mental-health-diagnosis-bert" 
repo_owner = "ethandavey"

# Create a full repository name
repo_id = f"{repo_owner}/{repo_name}" if repo_owner else repo_name

model_card = """---
language: en
license: mit
tags:
  - mental-health
  - clinical
  - bert
  - text-classification
widget:
  - text: "I've been feeling very sad and empty lately, and I don't enjoy things anymore."
---

# Mental Health Diagnosis BERT Model

This model fine-tunes Bio_ClinicalBERT to predict mental health diagnoses from patient statements. 
It can classify text into 5 categories:
- Anxiety
- Depression
- Suicidal
- Stress
- Normal

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("YOUR_USERNAME/mental-health-diagnosis-bert")
model = AutoModelForSequenceClassification.from_pretrained("YOUR_USERNAME/mental-health-diagnosis-bert")

# Prepare text
text = "I've been feeling very anxious and worried all the time."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

# Make prediction
with torch.no_grad():
    outputs = model(**inputs)
    probabilities = F.softmax(outputs.logits, dim=1)

# Map prediction to label
label_mapping = {0: "Anxiety", 1: "Normal", 2: "Depression", 3: "Suicidal", 4: "Stress"}
predicted_class = torch.argmax(probabilities, dim=1).item()
prediction = label_mapping[predicted_class]
confidence = probabilities[0][predicted_class].item()

print(f"Prediction: {prediction}, Confidence: {confidence:.2f}")
```
"""

# Create README.md with the model card
with open("README_huggingface.md", "w") as f:
    f.write(model_card)

# Step 5: Push to Hugging Face Hub
model.push_to_hub(
    repo_id=repo_id,
    commit_message="Upload fine-tuned Mental Health BERT model",
)

tokenizer.push_to_hub(
    repo_id=repo_id,
    commit_message="Upload tokenizer for Mental Health BERT model",
)

# Also push the README separately to make sure it's used as the model card
from huggingface_hub import upload_file
upload_file(
    path_or_fileobj="README_huggingface.md",
    path_in_repo="README.md",
    repo_id=repo_id,
)

print(f"Successfully uploaded model and tokenizer to {repo_id}")
print(f"Your model is now available at: https://huggingface.co/{repo_id}") 