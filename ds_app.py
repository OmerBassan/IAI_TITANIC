import streamlit as st
import pandas as pd
import torch
import joblib
from pathlib import Path
from src.model import TabularResNet

# --- הגדרות דף וטעינת מודלים ---
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="wide")

@st.cache_resource
def load_artifacts():
    """Loads the preprocessor and the trained PyTorch model."""
    # נטען את המודל המנצח מניסוי 8
    trial_dir = Path("artifacts/trial_8")
    
    preprocessor = joblib.load(trial_dir / "preprocessor.joblib")
    ckpt = torch.load(trial_dir / "model.pt", map_location="cpu")
    
    model = TabularResNet(**ckpt["arch"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    
    return preprocessor, model

try:
    preprocessor, model = load_artifacts()
except Exception as e:
    st.error(f"Failed to load model artifacts. Ensure 'artifacts/trial_8/' exists. Error: {e}")
    st.stop()

# --- עיצוב הממשק ---
st.title("🚢 Titanic Survival Predictor")
st.markdown("Enter passenger details below to predict their probability of surviving the Titanic disaster.")

# יצירת טופס להזנת הנתונים
with st.form("passenger_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Personal Info")
        name = st.text_input("Full Name (with title)", "Braund, Mr. Owen Harris")
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0)
        
    with col2:
        st.subheader("Ticket Details")
        pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2)
        ticket = st.text_input("Ticket Number", "A/5 21171")
        fare = st.number_input("Fare (£)", min_value=0.0, value=7.25)
        cabin = st.text_input("Cabin (Leave blank if unknown)", "")
        embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])
        
    with col3:
        st.subheader("Family Onboard")
        sibsp = st.number_input("Siblings / Spouses", min_value=0, max_value=10, value=0)
        parch = st.number_input("Parents / Children", min_value=0, max_value=10, value=0)

    submit_button = st.form_submit_button(label="Predict Survival")

# --- לוגיקת החיזוי ---
if submit_button:
    # 1. איסוף הקלט ל-DataFrame גולמי
    input_data = pd.DataFrame([{
        "PassengerId": 999, # לא באמת משנה לחיזוי
        "Pclass": pclass,
        "Name": name,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Ticket": ticket,
        "Fare": fare,
        "Cabin": cabin if cabin else float('nan'),
        "Embarked": embarked
    }])
    
    try:
        # 2. העברה ב-Preprocessor שלך
        x_transformed = preprocessor.transform(input_data)
        x_tensor = torch.tensor(x_transformed, dtype=torch.float32)
        
        # 3. הפעלת מודל PyTorch
        with torch.no_grad():
            proba = model.predict_proba(x_tensor).item()
            
        # 4. תצוגת התוצאה
        st.divider()
        if proba >= 0.5:
            st.success(f"### 🟢 Prediction: SURVIVED")
            st.info(f"Survival Probability: **{proba:.1%}**")
        else:
            st.error(f"### 🔴 Prediction: DID NOT SURVIVE")
            st.info(f"Survival Probability: **{proba:.1%}**")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")