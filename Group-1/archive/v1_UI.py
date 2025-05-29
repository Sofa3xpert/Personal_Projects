import gradio as gr
import sqlite3
import bcrypt
from datetime import datetime
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import os
import logging
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --------------------------
# Environment Setup
# --------------------------
  # Load environment variables from .env file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------
# Global Tokenizer and Model for Treatment (Hybrid Model)
# --------------------------
# Load tokenizer for hybrid model (used for medication & therapy prediction)
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")


# Define the Hybrid Model architecture
class HybridMentalHealthModel(nn.Module):
    def __init__(self, bert_model, num_genders, num_medications, num_therapies, hidden_size=128):
        super(HybridMentalHealthModel, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model)
        bert_output_size = self.bert.config.hidden_size
        # MLP layers for additional features
        self.age_fc = nn.Linear(1, 16)
        self.gender_fc = nn.Embedding(num_genders, 16)
        # Combined fully connected layer
        self.fc = nn.Linear(bert_output_size + 32, hidden_size)
        # Output heads for medication and therapy predictions
        self.medication_head = nn.Linear(hidden_size, num_medications)
        self.therapy_head = nn.Linear(hidden_size, num_therapies)

    def forward(self, input_ids, attention_mask, age, gender):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        age_out = self.age_fc(age)
        gender_out = self.gender_fc(gender)
        combined = torch.cat((bert_output, age_out, gender_out), dim=1)
        hidden = torch.relu(self.fc(combined))
        return self.medication_head(hidden), self.therapy_head(hidden)


# --------------------------
# Global Label Mappings and Age Scaler
# --------------------------
# These should match your training data
medication_classes = ["Anxiolytics", "Benzodiazepines", "Antidepressants", "Mood Stabilizers", "Antipsychotics",
                      "Stimulants"]
therapy_classes = ["TherapyA", "TherapyB", "TherapyC", "TherapyD"]  # Update with your actual therapy types
gender_classes = ["Male", "Female"]  # Update if necessary

# Create mapping dictionaries
medication_encoder = {name: idx for idx, name in enumerate(medication_classes)}
inv_medication_encoder = {idx: name for name, idx in medication_encoder.items()}
therapy_encoder = {name: idx for idx, name in enumerate(therapy_classes)}
inv_therapy_encoder = {idx: name for name, idx in therapy_encoder.items()}
gender_encoder = {name: idx for idx, name in enumerate(gender_classes)}

# For age normalization (update these values as per your training)
mean_age = 50
std_age = 10


def scale_age(age):
    return (age - mean_age) / std_age


# --------------------------
# Load the Hybrid Model (Treatment Prediction)
# --------------------------
num_genders = len(gender_classes)
num_medications = len(medication_classes)
num_therapies = len(therapy_classes)
MODEL_SAVE_PATH = "22.03.2025-16.02-ML128E10"  # Update with your saved model path

model = HybridMentalHealthModel("emilyalsentzer/Bio_ClinicalBERT", num_genders, num_medications, num_therapies)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model.to(device)
model.eval()

# --------------------------
# Global Diagnosis Model (Mental Health Diagnosis)
# --------------------------
# Load model and tokenizer for diagnosing mental health issues
diagnosis_tokenizer = AutoTokenizer.from_pretrained(
    "ethandavey/mental-health-diagnosis-bert")  # Update with your model ID
diagnosis_model = AutoModelForSequenceClassification.from_pretrained(
    "ethandavey/mental-health-diagnosis-bert")  # Update with your model ID
diagnosis_model.to(device)
diagnosis_model.eval()


def predict_disease(text):
    inputs = diagnosis_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = diagnosis_model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=1)
    label_mapping = {0: "Anxiety", 1: "Normal", 2: "Depression", 3: "Suicidal", 4: "Stress"}
    predicted_class = torch.argmax(probabilities, dim=1).item()
    prediction = label_mapping[predicted_class]
    confidence = probabilities[0][predicted_class].item()
    return prediction, confidence


def predict_med_therapy(symptoms, age, gender):
    # Tokenize the input text
    encoding = tokenizer(
        symptoms,
        return_tensors="pt",
        truncation=True,
        padding='max_length',
        max_length=128
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    # Normalize age and encode gender
    age_norm = torch.tensor([[scale_age(age)]], dtype=torch.float32).to(device)
    gender_idx = gender_encoder.get(gender, 0)
    gender_tensor = torch.tensor([gender_idx], dtype=torch.long).to(device)

    with torch.no_grad():
        med_logits, therapy_logits = model(input_ids, attention_mask, age_norm, gender_tensor)

    med_probabilities = torch.softmax(med_logits, dim=1)
    therapy_probabilities = torch.softmax(therapy_logits, dim=1)

    med_pred = torch.argmax(med_probabilities, dim=1).item()
    therapy_pred = torch.argmax(therapy_probabilities, dim=1).item()

    med_confidence = med_probabilities[0][med_pred].item()
    therapy_confidence = therapy_probabilities[0][therapy_pred].item()

    predicted_med = inv_medication_encoder.get(med_pred, "Unknown")
    predicted_therapy = inv_therapy_encoder.get(therapy_pred, "Unknown")
    return (predicted_med, med_confidence), (predicted_therapy, therapy_confidence)


# --------------------------
# OpenAI Functions (Summarization and Explanation)
# --------------------------
def get_concise_rewrite(text, max_tokens, temperature=0.7):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert rewriting assistant. Your task is to rewrite a long statement into a shorter version while "
                "preserving as much of the original vocabulary, tone, and style as possible. Do not include any phrases like "
                "'here is a summary:' or indicate that it is a summary. Simply produce a concise version of the statement as if "
                "the original author had written it more succinctly."
            )
        },
        {
            "role": "user",
            "content": text
        }
    ]
    try:
        response = client.chat.completions.create(model="gpt-4o-mini",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature)
        concise_text = response.choices[0].message.content.strip()
    except Exception as e:
        concise_text = f"API call failed: {e}"
    return concise_text


def get_explanation(patient_statement, predicted_diagnosis):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert mental health assistant. Your task is to provide a concise, evidence-based explanation of how a patient's statement supports a given diagnosis. Focus on the key aspects of the statement that indicate the diagnosis."
            )
        },
        {
            "role": "user",
            "content": (
                f"Patient statement: {patient_statement}\nPredicted diagnosis: {predicted_diagnosis}\nPlease provide a brief explanation of how the statement supports this diagnosis."
            )
        }
    ]
    try:
        response = client.chat.completions.create(model="gpt-4o-mini",
        messages=messages,
        max_tokens=256)
        explanation = response.choices[0].message.content.strip()
    except Exception as e:
        explanation = "API call failed."
    return explanation


# --------------------------
# Database Functions for Authentication and Chat History
# --------------------------
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        full_name TEXT,
        email TEXT
    )""")
    c.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        message TEXT NOT NULL,
        response TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()
    conn.close()


def register_user(username, password, full_name, email):
    if not re.fullmatch(r"[^@]+@[^@]+\.[^@]+", email):
        return "Invalid email format."
    if len(password) <= 8:
        return "Password must be more than 8 characters."
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    try:
        c.execute("INSERT INTO users (username, password, full_name, email) VALUES (?, ?, ?, ?)",
                  (username, hashed, full_name, email))
        conn.commit()
        return "User registered successfully."
    except sqlite3.IntegrityError:
        return "Username already exists."
    finally:
        conn.close()


def login_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    if user and bcrypt.checkpw(password.encode(), user[0]):
        return True
    return False


def get_chat_history(username):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT message, response, timestamp FROM chat_history WHERE username = ? ORDER BY timestamp DESC",
              (username,))
    history = c.fetchall()
    conn.close()
    return history


# --------------------------
# Gradio UI Setup
# --------------------------
with gr.Blocks(theme="soft") as app:
    # Shared state for user session
    user_session = gr.State(value="")

    # --- Login Panel (visible by default) ---
    with gr.Column(visible=True, elem_id="login_panel") as login_panel:
        gr.Markdown("## 🩺 Login / Register")
        with gr.Row():
            username_login = gr.Textbox(label="Username")
            password_login = gr.Textbox(label="Password", type="password")
        login_btn = gr.Button("Login")
        login_output = gr.Textbox(label="Login Status", interactive=False)

        with gr.Row():
            new_username = gr.Textbox(label="New Username")
            new_password = gr.Textbox(label="New Password", type="password")
            full_name = gr.Textbox(label="Full Name")
            email = gr.Textbox(label="Email")
        register_btn = gr.Button("Register")
        register_output = gr.Textbox(label="Registration Status", interactive=False)

    # --- Main Panel (hidden until login) ---
    with gr.Column(visible=False, elem_id="main_panel") as main_panel:
        gr.Markdown("## 🩺 Mental Health Chatbot")
        with gr.Tab("Chatbot"):
            with gr.Column():
                patient_name_input = gr.Textbox(placeholder="Enter your name", label="Patient Name")
                gender_input = gr.Dropdown(choices=list(gender_encoder.keys()), label="Gender")
                age_input = gr.Number(label="Age")
                symptoms_input = gr.Textbox(placeholder="Describe your symptoms", label="Symptoms", lines=4)
                submit = gr.Button("Submit")
            Diagnosis = gr.Textbox(label="Diagnosis", interactive=False)
            Treatment = gr.Textbox(label="Treatment", interactive=False)
            Summary = gr.Textbox(label="Concise Summary", interactive=False)
            Explanation = gr.Textbox(label="Explanation", interactive=False)


            def handle_chat_extended(patient_name, gender, age, symptoms):
                # Check input length
                if len(symptoms.split()) > 512:
                    msg = "Input exceeds maximum allowed length of 512 words."
                    return msg, msg, msg, msg
                full_statement = f"Patient Name: {patient_name}, Gender: {gender}, Age: {age}, Symptoms: {symptoms}"

                # Generate a concise summary
                summary = get_concise_rewrite(full_statement, max_tokens=150, temperature=0.7)

                # Predict mental health diagnosis with confidence
                diagnosis_pred, diagnosis_conf = predict_disease(full_statement)
                diagnosis_output = f"{diagnosis_pred} ({diagnosis_conf * 100:.1f}% confidence)"

                # Predict medication and therapy (with confidence scores)
                (med_pred, med_conf), (therapy_pred, therapy_conf) = predict_med_therapy(symptoms, age, gender)
                treatment_plan = f"Medication: {med_pred} ({med_conf * 100:.1f}% confidence); Therapy: {therapy_pred} ({therapy_conf * 100:.1f}% confidence)"

                # Generate an explanation using OpenAI
                explanation = get_explanation(full_statement, f"{diagnosis_pred}, {med_pred} and {therapy_pred}")

                # Save chat history if logged in
                conn = sqlite3.connect("users.db")
                c = conn.cursor()
                if user_session.value:
                    c.execute("INSERT INTO chat_history (username, message, response) VALUES (?, ?, ?)",
                              (user_session.value, full_statement, diagnosis_output))
                    conn.commit()
                conn.close()

                return diagnosis_output, treatment_plan, summary, explanation


            submit.click(handle_chat_extended,
                         inputs=[patient_name_input, gender_input, age_input, symptoms_input],
                         outputs=[Diagnosis, Treatment, Summary, Explanation])

        with gr.Tab("Chat History"):
            history_output = gr.Textbox(label="Chat History", lines=10)
            load_history_btn = gr.Button("Load History")


            def load_history():
                if user_session.value:
                    history = get_chat_history(user_session.value)
                    return "\n".join([f"[{h[2]}] {h[0]}\nBot: {h[1]}" for h in history])
                else:
                    return "Please log in to view history."


            load_history_btn.click(load_history, outputs=history_output)

        with gr.Tab("Book an Appointment"):
            with gr.Row():
                with gr.Column():
                    patient_name = gr.Textbox(label="Patient Name", placeholder="Enter your name")
                    doctor_name = gr.Dropdown(choices=["Dr. Smith", "Dr. Johnson", "Dr. Lee"], label="Select Doctor")
                    appointment_date = gr.Textbox(label="Appointment Date", placeholder="YYYY-MM-DD")
                    appointment_time = gr.Textbox(label="Appointment Time", placeholder="HH:MM (24-hour format)")
                    reason = gr.TextArea(label="Reason for Visit",
                                         placeholder="Describe your symptoms or reason for the visit")
                    book_button = gr.Button("Book Appointment")
                with gr.Column():
                    booking_output = gr.TextArea(label="Booking Confirmation", interactive=False)


                def book_appointment(patient_name, doctor_name, appointment_date, appointment_time, reason):
                    if not user_session.value:
                        return "Please log in to book an appointment."
                    patient_name = (patient_name or "").strip()
                    doctor_name = (doctor_name or "").strip()
                    appointment_date = (appointment_date or "").strip()
                    appointment_time = (appointment_time or "").strip()
                    reason = (reason or "").strip()
                    if not (patient_name and doctor_name and appointment_date and appointment_time and reason):
                        return "Please fill in all the fields before booking an appointment."
                    if not re.fullmatch(r"[A-Za-z ]+", patient_name):
                        return "Patient name should contain only letters and spaces."
                    try:
                        datetime.strptime(appointment_date, "%Y-%m-%d")
                    except ValueError:
                        return "Appointment date must be in the format YYYY-MM-DD."
                    try:
                        datetime.strptime(appointment_time, "%H:%M")
                    except ValueError:
                        return "Appointment time must be in the format HH:MM (24-hour)."
                    confirmation = (
                        f"Appointment booked for {patient_name} with {doctor_name} on "
                        f"{appointment_date} at {appointment_time}.\n\n"
                        f"**Reason:** {reason}"
                    )
                    return confirmation


                book_button.click(book_appointment,
                                  inputs=[patient_name, doctor_name, appointment_date, appointment_time, reason],
                                  outputs=booking_output)


    # --- Login/Register Handling ---
    def handle_login(username, password):
        if login_user(username, password):
            user_session.value = username
            # On successful login, reveal main panel and hide login panel
            return f"Welcome, {username}!", gr.update(visible=True), gr.update(visible=False)
        else:
            return "Invalid credentials.", gr.update(), gr.update()


    def handle_register(username, password, full_name, email):
        return register_user(username, password, full_name, email)


    login_btn.click(handle_login,
                    inputs=[username_login, password_login],
                    outputs=[login_output, main_panel, login_panel])
    register_btn.click(handle_register,
                       inputs=[new_username, new_password, full_name, email],
                       outputs=register_output)

init_db()
app.launch()
