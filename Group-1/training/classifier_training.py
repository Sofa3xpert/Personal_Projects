import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
from datetime import datetime
import optuna
import os

import wandb


# Define the label mapping
label_mapping = {"Anxiety": 0, "Normal": 1, "Depression": 2, "Suicidal": 3, "Stress": 4}

# Load data
train_df = pd.read_csv("./data/train_data.csv")
val_df = pd.read_csv("./data/val_data.csv")
test_df = pd.read_csv("./data/test_data.csv")

# Map the diagnosis labels to numerical labels using the "status" column
train_df["label"] = train_df["status"].map(label_mapping)
val_df["label"] = val_df["status"].map(label_mapping)
test_df["label"] = test_df["status"].map(label_mapping)

# Define constant hyperparameters
max_input_length = 256
num_epochs = 3

# Class for the dataset
class MentalHealthDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=max_input_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        # Squeeze to remove the extra batch dimension
        item = {key: encoding[key].squeeze() for key in encoding}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

# Load Bio_ClinicalBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# Create dataset objects for train, validation, and test sets using the "statement" column for text
train_dataset = MentalHealthDataset(train_df['statement'].tolist(), train_df['label'].tolist(), tokenizer)
val_dataset = MentalHealthDataset(val_df['statement'].tolist(), val_df['label'].tolist(), tokenizer)
test_dataset = MentalHealthDataset(test_df['statement'].tolist(), test_df['label'].tolist(), tokenizer)

# Define the objective function for Optuna
def objective(trial):
    # Initialize a new wandb run for this trial
    run = wandb.init(
        project="mental-health-classification",
        name=f"trial_{trial.number}",
        group="hyperparameter_search",
        reinit=True,
        config={
            "trial_number": trial.number,
            "max_input_length": max_input_length,
            "num_epochs": num_epochs,
        }
    )
    
    # Hyperparameters to optimize
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
    gradient_accumulation_steps = trial.suggest_categorical("gradient_accumulation_steps", [2, 4, 8])
    
    # Log hyperparameters to wandb
    wandb.config.update({
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    })
    
    # Create model for this trial
    model = AutoModelForSequenceClassification.from_pretrained(
        "emilyalsentzer/Bio_ClinicalBERT",
        num_labels=len(label_mapping)
    )
    
    # Define training arguments for this trial with wandb integration
    training_args = TrainingArguments(
        output_dir=f"./tmp/trial_{trial.number}",  # Temporary directory for checkpoints
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=None,  # Disable local logs
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",  # Enable wandb reporting
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    
    # Train the model
    trainer.train()
    
    # Evaluate on validation set
    eval_results = trainer.evaluate()
    
    # Close wandb run
    wandb.finish()
    
    # Return the validation loss as the metric to optimize
    return eval_results["eval_loss"]

# Initialize wandb for the entire hyperparameter search process
wandb.login()

# Create an Optuna study and optimize
study_name = "bert_hyperparameter_optimization_v2"
storage_name = "sqlite:///optuna_studies.db"

# Check if the study already exists
study = optuna.create_study(
    study_name=study_name,
    storage=storage_name,
    load_if_exists=True,
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
)

# Number of trials to run
n_trials = 20
print(f"Running optimization with {n_trials} trials...")
study.optimize(objective, n_trials=n_trials)

# Display the best parameters
print("Best trial:")
best_trial = study.best_trial
print(f"  Value (validation loss): {best_trial.value}")
print("  Params:")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

# Train the final model with the best hyperparameters
print("Training final model with best hyperparameters...")

# Start a new wandb run for the final model training
final_run = wandb.init(
    project="mental-health-classification",
    name="best_model_final_training",
    group="final_training",
    reinit=True,
    config={
        "max_input_length": max_input_length,
        "num_epochs": num_epochs,
        "learning_rate": best_trial.params["learning_rate"],
        "batch_size": best_trial.params["batch_size"],
        "weight_decay": best_trial.params["weight_decay"],
        "warmup_ratio": best_trial.params["warmup_ratio"],
        "gradient_accumulation_steps": best_trial.params["gradient_accumulation_steps"],
    }
)

# Create model
model = AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT",
                                                           num_labels=len(label_mapping))

# Define training arguments with best hyperparameters and wandb integration
final_training_args = TrainingArguments(
    output_dir="./tmp/best_model",  # Temporary directory for checkpoints
    num_train_epochs=num_epochs,
    per_device_train_batch_size=best_trial.params["batch_size"],
    per_device_eval_batch_size=best_trial.params["batch_size"],
    learning_rate=best_trial.params["learning_rate"],
    weight_decay=best_trial.params["weight_decay"],
    warmup_ratio=best_trial.params["warmup_ratio"],
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir=None,  # Disable local logs
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="wandb",  # Enable wandb reporting
    gradient_accumulation_steps=best_trial.params["gradient_accumulation_steps"],
)

# Initialize the Trainer
final_trainer = Trainer(
    model=model,
    args=final_training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# Start training
final_trainer.train()

# Evaluate on the test set after training
eval_results = final_trainer.evaluate(eval_dataset=test_dataset)
print("Test evaluation results:", eval_results)

# Log test results to wandb
wandb.log({"test_loss": eval_results["eval_loss"], "test_accuracy": eval_results.get("eval_accuracy", None)})

# Save the model named with a timestamp and hyperparameter configurations
current_time = datetime.now().strftime("%d.%m.%Y-%H.%M")
model_save_path = f"models/BERT/{current_time}-ML{max_input_length}E{num_epochs}"
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
final_trainer.save_model(model_save_path)
print(f"Model saved to {model_save_path}")

# Also save the best hyperparameters for reference
with open(f"{model_save_path}/best_hyperparameters.txt", "w") as f:
    f.write("Best hyperparameters:\n")
    for key, value in best_trial.params.items():
        f.write(f"{key}: {value}\n")

# Link the saved model to wandb
wandb.save(f"{model_save_path}/*")

# Finish the wandb run
wandb.finish()
