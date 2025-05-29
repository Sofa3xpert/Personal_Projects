# Mental Health Assistant Platform (ACADEMIC) Original Repo: https://github.kcl.ac.uk/k23051132/Group-1

A platform for mental health diagnosis, treatment recommendation, and patient management.

## Project Overview

This project integrates multiple AI models to assist with mental health diagnosis, medication and therapy recommendations, and patient management. The system provides:

1. **Mental Health Diagnosis**: Classification of patient statements into mental health conditions (Anxiety, Depression, Stress, Suicidal, Normal) --> https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health
2. **Treatment Recommendations**: Recommendation of medications and therapies based on patient symptoms and characteristics --> https://www.kaggle.com/datasets/uom190346a/mental-health-diagnosis-and-treatment-monitoring
3. **Patient Management UI**: A user-friendly interface for managing patient sessions, generating reports, and storing patient history
4. **LLM Integration**: Leveraging OpenAI's GPT models for explanation, text summarization, and further insights

## Features

- **User Authentication**: Secure login and registration system
- **Patient Session Management**: Record and manage patient sessions
- **Automated Diagnosis**: AI-powered diagnosis based on patient statements
- **Treatment Recommendations**: Personalized medication and therapy recommendations
- **PDF Report Generation**: Create professional reports for each patient session
- **Chat History**: Store and retrieve past patient interactions
- **Appointment Scheduling**: Book and manage patient appointments
  
## Deployment

- The project is deployed on [Hugging Face here](https://huggingface.co/spaces/Meshari21/AI_Project)


## Project Structure

```
.
├── UI/                   # User interface implementation with Gradio
├── OpenAI/               # OpenAI integration for LLM capabilities
│   ├── data/             # Data preparation for fine-tuning
│   ├── run/              # Inference scripts
│   └── training/         # Fine-tuning scripts for OpenAI models
├── HuggingFace/           # HuggingFace model upload and prediction scripts
├── Dataprep/             # Data preprocessing and preparation scripts
├── training/             # Model training scripts and notebooks
│   └── classifier_training.py  # BERT-based classifier training
├── unit-testing/         # Unit tests for the application
├── requirements.txt      # Project dependencies
└── .env                  # Environment variables (API keys, etc.)
```

## Models

1. **Mental Health Diagnosis Model**: Fine-tuned ClinicalBERT for mental health diagnosis classification
2. **Treatment Recommendation Model**: Hybrid model combining BERT embeddings with demographic features to recommend medications and therapies
3. **OpenAI GPT Models**: Used for text explanation and summarization

## Prerequisites

- **Python 3.8+** (Tested on Python 3.12)
- **CUDA-compatible GPU** (Optional but recommended for model training)
- **OpenAI API Key** (Required for LLM capabilities)
- **Weights & Biases Account** (Optional, for experiment tracking)

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/mental-health-assistant.git
   cd mental-health-assistant
   ```

2. **Set up a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a .env file with your OpenAI API key**
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. **Initialize the database**
   ```bash
   python -c "from UI.v4_UI_fix import init_db; init_db()"
   ```

## Running the Application

1. **Launch the UI**
   ```bash
   python UI/v4_UI_fix.py
   ```
   The application will be accessible at http://localhost:7860 by default.

2. **Register an account** and log in to access the patient management system.




