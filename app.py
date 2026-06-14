import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("Best_model.pkl")
encoders = joblib.load("encoders.pkl")

st.set_page_config(page_title="Employee Salary Prediction ", page_icon="💼", layout="centered")
st.title("💼Welcome,Employee Salary Prediction sute")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

st.sidebar.header("Input Employee Details")

# Collect inputs
age = st.sidebar.slider("Age", 18, 65, 30)
workclass = st.sidebar.selectbox("Workclass", encoders['workclass'].classes_)
fnlwgt = st.sidebar.number_input("FNLWGT", min_value=0, value=100000)
education_num = st.sidebar.slider("Education Num", 1, 16, 10)
marital_status = st.sidebar.selectbox("Marital Status", encoders['marital-status'].classes_)
occupation = st.sidebar.selectbox("Occupation", encoders['occupation'].classes_)
relationship = st.sidebar.selectbox("Relationship", encoders['relationship'].classes_)
race = st.sidebar.selectbox("Race", encoders['race'].classes_)
gender = st.sidebar.selectbox("Gender", encoders['gender'].classes_)
capital_gain = st.sidebar.number_input("Capital Gain", value=0)
capital_loss = st.sidebar.number_input("Capital Loss", value=0)
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
native_country = st.sidebar.selectbox("Country", encoders['native-country'].classes_)

# Encode inputs
input_dict = {
    'age': age,
    'workclass': encoders['workclass'].transform([workclass])[0],
    'fnlwgt': fnlwgt,
    'educational-num': education_num,
    'marital-status': encoders['marital-status'].transform([marital_status])[0],
    'occupation': encoders['occupation'].transform([occupation])[0],
    'relationship': encoders['relationship'].transform([relationship])[0],
    'race': encoders['race'].transform([race])[0],
    'gender': encoders['gender'].transform([gender])[0],
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'hours-per-week': hours_per_week,
    'native-country': encoders['native-country'].transform([native_country])[0]
}

input_df = pd.DataFrame([input_dict])

st.write("### 🔎 Input Data")
st.write(input_df)

if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f" Prediction: {prediction[0]}")

# Batch prediction
st.markdown("---")
st.markdown("####  Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)

    # Encode categorical columns
    for col in encoders:
        batch_data[col] = encoders[col].transform(batch_data[col])

    st.write("Uploaded data preview:", batch_data.head())
    batch_preds = model.predict(batch_data.drop('income', axis=1, errors='ignore'))
    batch_data['PredictedClass'] = batch_preds
    st.write("✅ Predictions:")
    st.write(batch_data.head())

    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
