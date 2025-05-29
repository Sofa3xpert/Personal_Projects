from datasets import load_dataset

# Load the dataset from Hugging Face
dataset = load_dataset("ruslanmv/ai-medical-chatbot", split="train")

# Save to local machine
dataset.to_parquet("../resources/dataset.parquet")

# Log info
print("Dataset downloaded and saved to resources/dataset.parquet")
