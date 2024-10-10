import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import time  # for creating progress bar animation

st.write("# 10 Year Heart Disease Prediction")

# Adding tooltips and visual descriptions with icons
col1, col2, col3 = st.columns(3)

# Getting user input with relevant icons
gender = col1.radio("Gender", ["Male", "Female"], help="Choose your gender")
age = col2.slider("Age", 20, 100, help="Enter your age (years)")
education = col3.selectbox("Highest Academic Qualification", 
                           ["High school diploma", "Undergraduate degree", "Postgraduate degree", "PhD"], 
                           help="Select your highest level of education")

isSmoker = col1.radio("🚬 Are you currently a smoker?", ["Yes", "No"], help="Do you smoke cigarettes regularly?")
yearsSmoking = col2.slider("Number of daily cigarettes", 0, 50, help="If you smoke, how many cigarettes do you smoke daily?")
BPMeds = col3.radio("💊 Are you currently on BP medication?", ["Yes", "No"], help="Are you taking any medication for blood pressure?")

stroke = col1.radio("🧠 Have you ever experienced a stroke?", ["Yes", "No"], help="Have you had any strokes in the past?")
hyp = col2.radio("🩺 Do you have hypertension?", ["Yes", "No"], help="Do you have high blood pressure?")
diabetes = col3.radio("🧪 Do you have diabetes?", ["Yes", "No"], help="Are you diagnosed with diabetes?")

# Intuitive numeric inputs with sliders
chol = col1.slider("Cholesterol Level (mg/dL)", 100, 400, help="Enter your cholesterol level")
sys_bp = col2.slider("Systolic Blood Pressure (mm Hg)", 90, 200, help="Systolic blood pressure (top number)")
dia_bp = col3.slider("Diastolic Blood Pressure (mm Hg)", 60, 130, help="Diastolic blood pressure (bottom number)")

bmi = col1.slider("BMI", 10.0, 50.0, step=0.1, help="Enter your Body Mass Index (BMI)")
heart_rate = col2.slider("Resting Heart Rate (bpm)", 40, 150, help="Resting heart rate (beats per minute)")
glucose = col3.slider("Glucose Level (mg/dL)", 60, 300, help="Enter your glucose level")

# Displaying a gauge chart or bar chart with animation using Plotly
st.write("### Your Health Metrics vs. Recommended Healthy Ranges")

# Data for user input and recommended healthy values
health_data = {
    'Cholesterol (mg/dL)': [chol, 125, 200],
    'Systolic BP (mm Hg)': [sys_bp, 90, 120],
    'Diastolic BP (mm Hg)': [dia_bp, 60, 80],
    'BMI': [bmi, 18.5, 24.9],
    'Resting Heart Rate (bpm)': [heart_rate, 60, 100],
    'Glucose (mg/dL)': [glucose, 70, 140]
}

# Create animated bar chart using frames
frames = []

# Initialize figure
fig = go.Figure()

# Add bars for each health metric (as frames)
for key, value in health_data.items():
    color = 'green' if value[0] < value[2] else 'yellow' if value[0] <= value[2] + 20 else 'red'
    
    # Create a frame for each health metric
    frames.append(go.Frame(
        data=[go.Bar(
            x=[key],
            y=[value[0]],
            marker_color=color,
            name=key,
            width=[0.4]
        )],
        name=key
    ))

# Initial frame to show the first value (can be empty)
fig.add_trace(go.Bar(x=[list(health_data.keys())[0]], y=[list(health_data.values())[0][0]], marker_color='green', width=0.4))

# Add frames to figure
fig.frames = frames

# Update layout for animation controls
fig.update_layout(
    title="Your Health Metrics vs. Healthy Ranges (Animated)",
    xaxis_title="Health Metrics",
    yaxis_title="Values",
    showlegend=False,
    updatemenus=[dict(type="buttons", showactive=False, 
                      buttons=[dict(label="Play",
                                    method="animate",
                                    args=[None, {"frame": {"duration": 1000, "redraw": True}, 
                                                 "fromcurrent": True, 
                                                 "transition": {"duration": 500}}])])]
)

st.plotly_chart(fig)

# Prepare data for prediction
df_pred = pd.DataFrame([[gender, age, education, isSmoker, yearsSmoking, BPMeds, stroke, hyp, diabetes, chol, sys_bp, dia_bp, bmi, heart_rate, glucose]],
                       columns=['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'])

# Preprocess the input data
df_pred['male'] = df_pred['male'].apply(lambda x: 1 if x == 'Male' else 0)
df_pred['currentSmoker'] = df_pred['currentSmoker'].apply(lambda x: 1 if x == 'Yes' else 0)
df_pred['prevalentStroke'] = df_pred['prevalentStroke'].apply(lambda x: 1 if x == 'Yes' else 0)
df_pred['prevalentHyp'] = df_pred['prevalentHyp'].apply(lambda x: 1 if x == 'Yes' else 0)
df_pred['diabetes'] = df_pred['diabetes'].apply(lambda x: 1 if x == 'Yes' else 0)
df_pred['BPMeds'] = df_pred['BPMeds'].apply(lambda x: 1 if x == 'Yes' else 0)

# Education transformation
def transform_education(data):
    if data == "High school diploma":
        return 0
    elif data == "Undergraduate degree":
        return 1
    elif data == "Postgraduate degree":
        return 2
    elif data == "PhD":
        return 3

df_pred['education'] = df_pred['education'].apply(transform_education)

# Load the model and make predictions
model = joblib.load('fhs_rf_model.pkl')

# Prediction with animated progress bar
if st.button('Predict'):
    st.write("Predicting...")

    # Animated progress bar to simulate loading
    progress = st.progress(0)
    for percent in range(0, 101, 5):
        time.sleep(0.1)
        progress.progress(percent)

    # Display the result with a final animated progress bar
    prediction = model.predict(df_pred)
    
    if prediction[0] == 0:
        st.success('You are unlikely to develop heart disease in the next 10 years.')
        st.write("Keep maintaining a healthy lifestyle!")
        
        # Final progress bar animation for low-risk
        for percent in range(0, 26, 5):
            time.sleep(0.1)
            st.progress(percent)
    else:
        st.error('You are at risk of developing heart disease in the next 10 years.')
        st.write("Please consult a healthcare provider for personalized advice.")
        
        # Final progress bar animation for high-risk
        for percent in range(75, 101, 5):
            time.sleep(0.1)
            st.progress(percent)
