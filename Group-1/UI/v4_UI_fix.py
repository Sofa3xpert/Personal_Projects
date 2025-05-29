import gradio as gr
import sqlite3
import bcrypt
from datetime import datetime
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()  # Loads .env file
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
import json
from fpdf import FPDF

# --------------------------
# Environment Setup
# --------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------
# Global Tokenizer and Hybrid Model for Treatment Prediction
# --------------------------
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")


class HybridMentalHealthModel(nn.Module):
    def __init__(self, bert_model, num_genders, num_medications, num_therapies, hidden_size=128):
        super(HybridMentalHealthModel, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model)
        bert_output_size = self.bert.config.hidden_size
        self.age_fc = nn.Linear(1, 16)
        self.gender_fc = nn.Embedding(num_genders, 16)
        self.fc = nn.Linear(bert_output_size + 32, hidden_size)
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
medication_classes = ["Anxiolytics", "Benzodiazepines", "Antidepressants", "Mood Stabilizers", "Antipsychotics", "Stimulants"]
therapy_classes = ["Cognitive Behavioral Therapy", "Dialectical Behavioral Therapy", "Interpersonal Therapy", "Mindfulness-Based Therapy"]  # Update with your types
gender_classes = ["Male", "Female", "Other"]

medication_encoder = {name: idx for idx, name in enumerate(medication_classes)}
inv_medication_encoder = {idx: name for name, idx in medication_encoder.items()}
therapy_encoder = {name: idx for idx, name in enumerate(therapy_classes)}
inv_therapy_encoder = {idx: name for name, idx in therapy_encoder.items()}
gender_encoder = {name: idx for idx, name in enumerate(gender_classes)}

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
MODEL_SAVE_PATH = "22.03.2025-16.02-ML128E10"  # Update accordingly

model = HybridMentalHealthModel("emilyalsentzer/Bio_ClinicalBERT", num_genders, num_medications, num_therapies)
state_dict = torch.load(MODEL_SAVE_PATH, map_location=device)
if "gender_fc.weight" in state_dict:
    del state_dict["gender_fc.weight"]
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()

# --------------------------
# Global Diagnosis Model (Mental Health Diagnosis)
# --------------------------
diagnosis_tokenizer = AutoTokenizer.from_pretrained("ethandavey/mental-health-diagnosis-bert")  # Update with your model ID
diagnosis_model = AutoModelForSequenceClassification.from_pretrained("ethandavey/mental-health-diagnosis-bert")  # Update with your model ID
diagnosis_model.to(device)
diagnosis_model.eval()

def predict_disease(text):
    inputs = diagnosis_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = diagnosis_model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=1).squeeze()
    label_mapping = {0: "Anxiety", 1: "Normal", 2: "Depression", 3: "Suicidal", 4: "Stress"}

    topk = torch.topk(probabilities, k=3)
    top_preds = [(label_mapping[i.item()], probabilities[i].item()) for i in topk.indices]
    return top_preds


def predict_med_therapy(symptoms, age, gender):
    encoding = tokenizer(symptoms, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
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
        {"role": "system", "content": "You are an expert rewriting assistant. Rewrite the given statement into a concise version while preserving its tone and vocabulary."},
        {"role": "user", "content": text}
    ]
    try:
        response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, max_tokens=max_tokens, temperature=temperature)
        concise_text = response.choices[0].message.content.strip()
    except Exception as e:
        concise_text = f"API call failed: {e}"
    return concise_text

def get_explanation(patient_statement, predicted_diagnosis):
    messages = [
        {"role": "system", "content": "You are an expert mental health assistant. Provide a concise, evidence-based explanation of how the patient's statement supports the diagnosis."},
        {"role": "user", "content": f"Patient statement: {patient_statement}\nPredicted diagnosis: {predicted_diagnosis}\nExplain briefly."}
    ]
    try:
        response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, max_tokens=256)
        explanation = response.choices[0].message.content.strip()
    except Exception as e:
        explanation = "API call failed."
    return explanation

# --------------------------
# Database Functions
# --------------------------
def init_db():
    conn = sqlite3.connect("../users.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            full_name TEXT,
            email TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            message TEXT NOT NULL,
            response TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS patient_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            patient_name TEXT,
            age REAL,
            gender TEXT,
            symptoms TEXT,
            diagnosis TEXT,
            medication TEXT,
            therapy TEXT,
            summary TEXT,
            explanation TEXT,
            pdf_report TEXT,
            session_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            appointment_date DATE
        )
    """)
    conn.commit()
    conn.close()

def register_user(username, password, full_name, email):
    if not re.fullmatch(r"[^@]+@[^@]+\.[^@]+", email):
        return "Invalid email format."
    if len(password) <= 8:
        return "Password must be more than 8 characters."
    conn = sqlite3.connect("../users.db")
    c = conn.cursor()
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    try:
        c.execute("INSERT INTO users (username, password, full_name, email) VALUES (?, ?, ?, ?)", (username, hashed, full_name, email))
        conn.commit()
        return "User registered successfully."
    except sqlite3.IntegrityError:
        return "Username already exists."
    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect("../users.db")
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    if user and bcrypt.checkpw(password.encode(), user[0]):
        return True
    return False

def get_user_info(username):
    conn = sqlite3.connect("../users.db")
    c = conn.cursor()
    c.execute("SELECT username, email, full_name FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    if user:
        return f"Username: {user[0]}\nFull Name: {user[2]}\nEmail: {user[1]}"
    else:
        return "User info not found."

def get_chat_history(username):
    conn = sqlite3.connect("../users.db")
    c = conn.cursor()
    c.execute("SELECT message, response, timestamp FROM chat_history WHERE username = ? ORDER BY timestamp DESC", (username,))
    history = c.fetchall()
    conn.close()
    return history

def get_patient_sessions(filter_name="", filter_date=""):
    conn = sqlite3.connect("../users.db")
    c = conn.cursor()
    query = "SELECT patient_name, age, gender, symptoms, diagnosis, medication, therapy, summary, explanation, pdf_report, session_timestamp FROM patient_sessions WHERE 1=1"
    params = []
    if filter_name:
        query += " AND patient_name LIKE ?"
        params.append(f"%{filter_name}%")
    if filter_date:
        query += " AND DATE(session_timestamp)=?"
        params.append(filter_date)
    c.execute(query, params)
    sessions = c.fetchall()
    conn.close()
    return sessions

def insert_patient_session(session_data):
    conn = sqlite3.connect("../users.db")
    c = conn.cursor()
    c.execute("""
        INSERT INTO patient_sessions (username, patient_name, age, gender, symptoms, diagnosis, medication, therapy, summary, explanation, pdf_report, appointment_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session_data.get("username"), session_data.get("patient_name"), session_data.get("age"), session_data.get("gender"),
        session_data.get("symptoms"), session_data.get("diagnosis"), session_data.get("medication"),
        session_data.get("therapy"), session_data.get("summary"), session_data.get("explanation"),
        session_data.get("pdf_report"), session_data.get("appointment_date")))
    conn.commit()
    conn.close()

# --------------------------
# PDF Report Generation Function
# --------------------------
def generate_pdf_report(session_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Patient Session Report", ln=True, align='C')
    pdf.ln(10)
    for key, value in session_data.items():
        pdf.multi_cell(0, 10, txt=f"{key.capitalize()}: {value}")
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{reports_dir}/{session_data.get('patient_name')}_{timestamp}.pdf"
    pdf.output(filename)
    return filename

# --------------------------
# Helper: Autofill Previous Patient Info
# --------------------------
def get_previous_patient_info(selected_patient):
    conn = sqlite3.connect("../users.db")
    c = conn.cursor()
    c.execute("SELECT patient_name, age, gender FROM patient_sessions WHERE patient_name=? ORDER BY session_timestamp DESC LIMIT 1", (selected_patient,))
    record = c.fetchone()
    conn.close()
    if record:
        return record[0], record[1], record[2]
    else:
        return "", None, ""

def get_previous_patients():
    conn = sqlite3.connect("../users.db")
    c = conn.cursor()
    c.execute("SELECT DISTINCT patient_name FROM patient_sessions")
    records = c.fetchall()
    conn.close()
    return [r[0] for r in records]

# --------------------------
# Gradio UI Setup with External CSS
# --------------------------
with gr.Blocks(css=open("styles.css", "r").read(), theme="soft") as app:
    user_session = gr.State(value="")
    profile_visible = gr.State(value=False)
    session_data_state = gr.State(value="")

    with gr.Row(elem_id="header") as header_row:
        with gr.Column(scale=8):
            gr.Markdown("## Mental Health Chatbot")
        with gr.Column(scale=4) as profile_container:
            profile_button = gr.Button("👤", elem_id="profile_button", variant="secondary")
            with gr.Column(visible=False, elem_id="profile_info_box") as profile_info_box:
                profile_info = gr.HTML()
                logout_button = gr.Button("Logout", elem_id="logout_button")

    with gr.Column(visible=True, elem_id="login_page") as login_page:
        gr.Markdown("## Login")
        with gr.Row():
            username_login = gr.Textbox(label="Username")
            password_login = gr.Textbox(label="Password", type="password")
        login_btn = gr.Button("Login")
        login_output = gr.Textbox(label="Login Status", interactive=False)
        gr.Markdown("New user? Click below to register.")
        go_to_register = gr.Button("Go to Register")

    with gr.Column(visible=False, elem_id="register_page") as register_page:
        gr.Markdown("## Register")
        new_username = gr.Textbox(label="New Username")
        new_password = gr.Textbox(label="New Password", type="password")
        full_name = gr.Textbox(label="Full Name")
        email = gr.Textbox(label="Email")
        register_btn = gr.Button("Register")
        register_output = gr.Textbox(label="Registration Status", interactive=False)
        gr.Markdown("Already have an account?")
        back_to_login = gr.Button("Back to Login")

    with gr.Tabs(visible=False, elem_id="main_panel") as main_panel:
        with gr.Tab("Chatbot"):
            with gr.Row():
                with gr.Column(scale=1):
                    previous_patient = gr.Dropdown(label="Previous Patients", choices=[], interactive=True)
                    patient_name_input = gr.Textbox(placeholder="Enter patient name", label="Patient Name")
                    gender_input = gr.Dropdown(choices=list(gender_encoder.keys()), label="Gender")
                    age_input = gr.Number(label="Age")
                    symptoms_input = gr.Textbox(placeholder="Describe symptoms", label="Symptoms", lines=4)
                    submit = gr.Button("Submit")
                    generate_report_btn = gr.Button("Generate Report", visible=False)
                with gr.Column(scale=1):
                    with gr.Row():
                        with gr.Column(scale=4, min_width=240):  # Textbox column
                            diagnosis_textbox = gr.Textbox(label="Diagnosis", 
                                                        interactive=False)
                        with gr.Column(scale=1, min_width=120):  # Confidence column
                            diagnosis_conf_html = gr.HTML(elem_classes=["confidence-container"])

                    with gr.Row():
                        with gr.Column(scale=4, min_width=240):
                            medication_textbox = gr.Textbox(label="Medication", 
                                                        interactive=False)
                        with gr.Column(scale=1, min_width=120):
                            medication_conf_html = gr.HTML(elem_classes=["confidence-container"])

                    with gr.Row():
                        with gr.Column(scale=4, min_width=240):
                            therapy_textbox = gr.Textbox(label="Therapy", 
                                                    interactive=False)
                        with gr.Column(scale=1, min_width=120):
                            therapy_conf_html = gr.HTML(elem_classes=["confidence-container"])
                    summary_textbox = gr.Textbox(label="Concise Summary", interactive=False)
                    explanation_textbox = gr.Textbox(label="Explanation", interactive=False)
            with gr.Row():
                report_download = gr.File(label="Download Report", interactive=False)

            def handle_chat_extended(patient_name, gender, age, symptoms):
                if age is None or age <= 0:
                    error_msg = "Age must be greater than 0."
                    return (error_msg, "", error_msg, "", error_msg, "", error_msg, error_msg, gr.update(visible=False))
                
                if age > 150:
                    error_msg2 = "Age must be lower than 150"
                    return (error_msg2, "", error_msg2, "", error_msg2, "", error_msg2, error_msg2, gr.update(visible=False))
                
                if len(symptoms.split()) > 512:
                    msg = "Input exceeds maximum allowed length of 512 words."
                    return (msg, "", msg, "", msg, "", msg, msg, gr.update(visible=False))

                full_statement = f"Patient Name: {patient_name}, Gender: {gender}, Age: {age}, Symptoms: {symptoms}"
                summary = get_concise_rewrite(full_statement, max_tokens=150, temperature=0.7)

                # Predict top 3 diagnoses
                diagnosis_preds = predict_disease(full_statement)  # Now returns list of (label, confidence)
                diagnosis_display = "\n".join([f"{label}" for label, _ in diagnosis_preds])

                def get_confidence_class(percentage):
                    if percentage <= 50:
                        return "confidence-low"
                    elif percentage <= 74:
                        return "confidence-medium"
                    else:
                        return "confidence-high"

                diagnosis_conf_html_val = "<div class='confidence-multi'>" + "<br>".join([
                    f"<div class='confidence-display'><span class='confidence-value {get_confidence_class(conf * 100)}'>{conf * 100:.1f}% confidence</span></div>"
                    for _, conf in diagnosis_preds
                ]) + "</div>"

                # Predict medication and therapy
                (med_pred, med_conf), (therapy_pred, therapy_conf) = predict_med_therapy(symptoms, age, gender)
                med_percentage = med_conf * 100
                therapy_percentage = therapy_conf * 100

                def get_conf_html(percentage):
                    return f"""
                    <div class="confidence-display">
                        <span class="confidence-value {get_confidence_class(percentage)}">
                            {percentage:.1f}% confidence
                        </span>
                    </div>
                    """

                medication_conf_html_val = get_conf_html(med_percentage)
                therapy_conf_html_val = get_conf_html(therapy_percentage)

                # Explanation
                top_diag_labels = ", ".join([label for label, _ in diagnosis_preds])
                explanation = get_explanation(full_statement, f"{top_diag_labels}, {med_pred} and {therapy_pred}")

                # Prepare session data
                top_diag_with_conf = ", ".join([f"{label} ({conf * 100:.1f}%)" for label, conf in diagnosis_preds])
                session_data = {
                    "patient_name": patient_name,
                    "age": age,
                    "gender": gender,
                    "symptoms": symptoms,
                    "diagnosis": top_diag_with_conf,
                    "medication": f"{med_pred} ({med_percentage:.1f}% confidence)",
                    "therapy": f"{therapy_pred} ({therapy_percentage:.1f}% confidence)",
                    "summary": summary,
                    "explanation": explanation,
                    "session_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                session_data_state.value = json.dumps(session_data)

                # Save to chat history
                conn = sqlite3.connect("../users.db")
                c = conn.cursor()
                if user_session.value:
                    c.execute("INSERT INTO chat_history (username, message, response) VALUES (?, ?, ?)",
                            (user_session.value, full_statement, top_diag_with_conf))
                    conn.commit()
                conn.close()

                return (
                    diagnosis_display, diagnosis_conf_html_val,
                    med_pred, medication_conf_html_val,
                    therapy_pred, therapy_conf_html_val,
                    summary, explanation,
                    gr.update(visible=True)
                )


            submit.click(handle_chat_extended,
                         inputs=[patient_name_input, gender_input, age_input, symptoms_input],
                         outputs=[diagnosis_textbox, diagnosis_conf_html, medication_textbox, medication_conf_html,
                                  therapy_textbox, therapy_conf_html, summary_textbox, explanation_textbox,
                                  generate_report_btn])

            def handle_generate_report():
                try:
                    data = json.loads(session_data_state.value)
                except:
                    return None
                pdf_file = generate_pdf_report(data)
                data["username"] = user_session.value
                data["appointment_date"] = ""
                data["pdf_report"] = pdf_file
                insert_patient_session(data)
                return pdf_file

            generate_report_btn.click(handle_generate_report, outputs=report_download)

            def autofill_previous(selected_patient):
                name, age_val, gender_val = get_previous_patient_info(selected_patient)
                return name, age_val, gender_val

            previous_patient.change(autofill_previous,
                                    inputs=[previous_patient],
                                    outputs=[patient_name_input, age_input, gender_input])

        with gr.Tab("Chat History"):
            history_output = gr.Textbox(label="Chat History", interactive=False)
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
                    patient_name_appt = gr.Textbox(label="Patient Name", placeholder="Enter your name")
                    doctor_name = gr.Dropdown(choices=["Dr. Smith", "Dr. Johnson", "Dr. Lee"], label="Select Doctor")
                    appointment_date = gr.Textbox(label="Appointment Date", placeholder="YYYY-MM-DD")
                    appointment_time = gr.Textbox(label="Appointment Time", placeholder="HH:MM (24-hour format)")
                    reason = gr.TextArea(label="Reason for Visit", placeholder="Describe your symptoms or reason for the visit")
                    book_button = gr.Button("Book Appointment")
                with gr.Column():
                    booking_output = gr.Textbox(label="Booking Confirmation", interactive=False)

            def book_appointment(patient_name, doctor_name, appointment_date, appointment_time, reason):
                if not user_session.value:
                    return "Please log in to book an appointment."
                patient_name = (patient_name or "").strip()
                doctor_name = (doctor_name or "").strip()
                appointment_date = (appointment_date or "").strip()
                appointment_time = (appointment_time or "").strip()
                reason = (reason or "").strip()
                if not (patient_name and doctor_name and appointment_date and appointment_time and reason):
                    return "Please fill in all the fields."
                if not re.fullmatch(r"[A-Za-z ]+", patient_name):
                    return "Patient name should contain only letters and spaces."
                try:
                    datetime.strptime(appointment_date, "%Y-%m-%d")
                except ValueError:
                    return "Appointment date must be in YYYY-MM-DD format."
                try:
                    datetime.strptime(appointment_time, "%H:%M")
                except ValueError:
                    return "Appointment time must be in HH:MM (24-hour) format."
                confirmation = (f"Appointment booked for {patient_name} with {doctor_name} on {appointment_date} at {appointment_time}.\n\n"
                                f"Reason: {reason}")
                return confirmation

            book_button.click(book_appointment,
                              inputs=[patient_name_appt, doctor_name, appointment_date, appointment_time, reason],
                              outputs=booking_output)

        with gr.Tab("Patient Sessions"):
            gr.Markdown("### Search Patient Sessions")
            search_name = gr.Textbox(label="Patient Name (optional)")
            search_date = gr.Textbox(label="Date (YYYY-MM-DD, optional)")
            search_button = gr.Button("Search")
            sessions_output = gr.Textbox(label="Sessions", interactive=False)

            def search_sessions(name, date):
                sessions = get_patient_sessions(filter_name=name, filter_date=date)
                if sessions:
                    output = "\n\n".join([f"Patient: {s[0]}\nAge: {s[1]}\nGender: {s[2]}\nSymptoms: {s[3]}\nDiagnosis: {s[4]}\nMedication: {s[5]}\nTherapy: {s[6]}\nSummary: {s[7]}\nExplanation: {s[8]}\nReport: {s[9]}\nSession Time: {s[10]}" for s in sessions])
                    return output
                else:
                    return "No sessions found."

            search_button.click(search_sessions, inputs=[search_name, search_date], outputs=sessions_output)

    def handle_login(username, password):
        if login_user(username, password):
            user_session.value = username
            prev_choices = get_previous_patients()
            return (f"Welcome, {username}!",
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(choices=prev_choices))
        else:
            return "Invalid credentials.", gr.update(), gr.update(), gr.update(), gr.update()

    def handle_register(username, password, full_name, email):
        return register_user(username, password, full_name, email)

    def go_to_register_page():
        return gr.update(visible=False), gr.update(visible=True)

    def back_to_login_page():
        return gr.update(visible=True), gr.update(visible=False)

    login_btn.click(handle_login,
                    inputs=[username_login, password_login],
                    outputs=[login_output, main_panel, login_page, header_row])
    go_to_register.click(go_to_register_page, outputs=[login_page, register_page])
    register_btn.click(handle_register,
                       inputs=[new_username, new_password, full_name, email],
                       outputs=register_output)
    back_to_login.click(back_to_login_page, outputs=[login_page, register_page])


    # Toggle profile function
    def toggle_profile(user, current_visible):
        if not user:
            return gr.update(visible=False), False, ""
        new_visible = not current_visible
        info = get_user_info(user) if new_visible else ""
        return gr.update(visible=new_visible), new_visible, info


    # Connect profile button click with correct input order:
    profile_button.click(
        toggle_profile,
        inputs=[user_session, profile_visible],
        outputs=[profile_info_box, profile_visible, profile_info]
    )


    # Handle login: update previous patients dropdown
    def handle_login(username, password):
        if login_user(username, password):
            user_session.value = username
            prev_choices = get_previous_patients()
            return (f"Welcome, {username}!",
                    gr.update(visible=True),  # main_panel visible
                    gr.update(visible=False),  # login_page hidden
                    gr.update(visible=True),  # header_row visible
                    gr.update(choices=prev_choices))  # update dropdown choices
        else:
            return "Invalid credentials.", gr.update(), gr.update(), gr.update(), gr.update()


    # Connect login button click:
    login_btn.click(
        handle_login,
        inputs=[username_login, password_login],
        outputs=[login_output, main_panel, login_page, header_row, previous_patient]
    )

init_db()
app.launch()
