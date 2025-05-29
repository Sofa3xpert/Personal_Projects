import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
import pandas as pd
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def predict_single_statement(text, model, tokenizer):
    """Predict a mental health diagnosis for a single text statement."""
    # Define the label mapping
    label_mapping = {0: "Anxiety", 1: "Normal", 2: "Depression", 3: "Suicidal", 4: "Stress"}
    
    # Tokenize the statement
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding='max_length',
        max_length=512
    )

    # Get prediction
    with torch.no_grad():
        outputs = model(**encoding)
    
    # Convert logits to probabilities using softmax
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1)
    
    # Get top 3 predictions
    top3_prob, top3_idx = torch.topk(probabilities, k=3, dim=1)
    top3_prob = top3_prob.squeeze()  # Tensor of shape (3,)
    top3_idx = top3_idx.squeeze()  # Tensor of shape (3,)
    
    result = []
    for i in range(3):
        diagnosis = label_mapping[top3_idx[i].item()]
        confidence = top3_prob[i].item()
        result.append((diagnosis, confidence))
    
    return result


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Predict mental health diagnoses from text statements')
    parser.add_argument('--model_id', type=str, default="ethandavey/mental-health-diagnosis-bert",
                        help='Hugging Face model ID')
    parser.add_argument('--text', type=str, help='Single text statement to predict')

    
    args = parser.parse_args()

    
    # Load model and tokenizer
    logging.info(f"Loading model from {args.model_id}...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_id)
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        logging.info("If this is the first time using Hugging Face, you may need to run: "
                    "huggingface-cli login")
        return
    
    # Put the model in evaluation mode
    model.eval()
    
    # Process single text
    if args.text:
        logging.info("Processing statement...")
        results = predict_single_statement(args.text, model, tokenizer)
        
        print("\nPrediction Results:")
        for i, (diagnosis, confidence) in enumerate(results, 1):
            print(f"{i}. {diagnosis}: {confidence:.2f}")


if __name__ == "__main__":
    main() 