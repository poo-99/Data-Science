import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Load Model and Preprocessor ---
# Load the files saved after model training
try:
    # IMPORTANT: These files must exist in the same directory as app.py
    model = joblib.load('logistic_regression_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'logistic_regression_model.pkl' and 'preprocessor.pkl' are in the same directory.")
    st.stop()

# --- Streamlit App Configuration ---
st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

st.title("🚢 Titanic Survival Prediction (Logistic Regression)")
st.write("Enter the passenger details to predict the survival probability.")
st.markdown("---")

# --- User Input Fields ---
with st.form("prediction_form"):
    st.header("Passenger Details")

    # Pclass
    pclass = st.selectbox('Passenger Class (Pclass)', options=[1, 2, 3], index=2)

    # Sex
    sex = st.selectbox('Sex', options=['male', 'female'], index=0)

    # Age
    age = st.slider('Age', min_value=0.42, max_value=80.0, value=30.0, step=1.0)

    # Fare
    fare = st.slider('Fare ($)', min_value=0.0, max_value=512.0, value=30.0, step=5.0)

    # SibSp
    sibsp = st.number_input('Number of Siblings/Spouses Aboard (SibSp)', min_value=0, max_value=8, value=0)

    # Parch
    parch = st.number_input('Number of Parents/Children Aboard (Parch)', min_value=0, max_value=6, value=0)

    # Embarked
    embarked = st.selectbox('Port of Embarkation', options=['S', 'C', 'Q'], index=0)

    submit_button = st.form_submit_button(label='Predict Survival')


# --- Prediction Logic ---
if submit_button:
    # 1. Create a DataFrame from user input
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked]
    })

    # 2. Apply the preprocessor (CRITICAL STEP)
    # This transforms the user's data exactly as the training data was transformed
    try:
        input_processed = preprocessor.transform(input_data)
    except Exception as e:
        st.error(f"Error during data processing: {e}")
        st.stop()

    # 3. Get prediction and probability
    probability_survival = model.predict_proba(input_processed)[0, 1]
    prediction = model.predict(input_processed)[0]

    # 4. Display Results
    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(f"**Prediction: Likely survived!**")
    else:
        st.error(f"**Prediction: Likely did not survive.**")

    st.info(f"The model predicts a **{probability_survival*100:.2f}%** chance of survival.")
