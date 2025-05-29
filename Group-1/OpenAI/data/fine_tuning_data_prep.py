import pandas as pd
import json

# Load the dataset and drop Description
df = pd.read_parquet("../resources/dataset.parquet")
df = df.drop(columns=["Description"])

# Define the system prompt
system_prompt = "You are a helpful AI medical chatbot."

# Function to clean non-breaking spaces found in downloaded dataset
def clean_text(text):
    return text.replace("\xa0", " ").strip() if isinstance(text, str) else text

# Create a dataframe of the rows to be used for training
train_df = df.iloc[200:250]
# Create train array to store JSONL data
train_jsonl = []

# Write each row into Chat Completions format
for _, row in train_df.iterrows():
    entry = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": clean_text(row["Patient"])},
            {"role": "assistant", "content": clean_text(row["Doctor"])}
        ]
    }
    # Add to train array
    train_jsonl.append(entry)

# Write the training JSONL
with open("../resources/train.jsonl", "w", encoding="utf-8") as f:
    for entry in train_jsonl:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# Log info
print("Saved training file to resources/train.jsonl.")

# Create a dataframe of the rows to be used for validation
val_df = df.iloc[250:260]
# Create validation array to store JSONL data
val_jsonl = []

# Write each row into Chat Completions format
for _, row in val_df.iterrows():
    entry = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": clean_text(row["Patient"])},
            {"role": "assistant", "content": clean_text(row["Doctor"])}
        ]
    }
    # Add to validation array
    val_jsonl.append(entry)

# Write the validation JSONL
with open("../resources/val.jsonl", "w", encoding="utf-8") as f:
    for entry in val_jsonl:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# Log info
print("Saved validation file to resources/val.jsonl.")
