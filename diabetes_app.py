import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# --- PAGE CONFIG ---
st.set_page_config(page_title="Diabetes Prediction App", layout="centered")

# --- LOAD AND TRAIN MODEL ---
# In a real-world scenario, you would save/load a pickle file (.pkl). 
# Here, we follow the notebook's logic by training on the fly.
@st.cache_resource
def train_model():
    # Load dataset
    try:
        df = pd.read_csv('diabetes.csv')
    except FileNotFoundError:
        st.error("Dataset 'diabetes.csv' not found. Please ensure it is in the same directory.")
        return None

    # Splitting features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Train test split (following general notebook structure)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

model = train_model()

# --- STREAMLIT UI ---
st.title("🩺 Diabetes Prediction Tool")
st.write("Enter the patient's details below to predict the likelihood of diabetes.")

if model:
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
            glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=100)
            blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=150, value=70)
            skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
            
        with col2:
            insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=80)
            bmi = st.number_input("BMI (Weight in kg/(height in m)^2)", min_value=0.0, max_value=70.0, value=25.0)
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
            age = st.number_input("Age", min_value=1, max_value=120, value=30)

        submit = st.form_submit_button("Predict Result")

    if submit:
        # Prepare data for prediction
        user_data = pd.DataFrame([[
            pregnancies, glucose, blood_pressure, skin_thickness, 
            insulin, bmi, dpf, age
        ]], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
        
        # Make prediction
        prediction = model.predict(user_data)
        probability = model.predict_proba(user_data)[0][1]

        # Display results
        st.divider()
        if prediction[0] == 1:
            st.warning(f"Result: Likely to have Diabetes (Probability: {probability:.2%})")
        else:
            st.success(f"Result: Unlikely to have Diabetes (Probability: {probability:.2%})")

else:
    st.info("Please upload 'diabetes.csv' to enable the model.")