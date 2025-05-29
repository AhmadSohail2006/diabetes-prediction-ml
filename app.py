import gradio as gr
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load your trained model (replace with your actual model file)
model = joblib.load('model.pkl')  
scaler = joblib.load('scaler.pkl')  # If you saved your scaler

def predict_diabetes(glucose, bmi, age, pregnancies, skin_thickness, insulin, bp, dpf):
    # Scale input features
    input_data = scaler.transform([[glucose, bmi, age, pregnancies, skin_thickness, insulin, bp, dpf]])
    
    # Get prediction probability
    proba = model.predict_proba(input_data)[0][1]
    
    return {"Diabetic": float(proba), "Healthy": 1-float(proba)}

# Create the interface
inputs = [
    gr.Slider(0, 200, label="Glucose Level"),
    gr.Slider(10, 50, label="BMI"),
    gr.Slider(20, 100, label="Age"),
    gr.Slider(0, 20, label="Pregnancies"),
    gr.Slider(0, 100, label="Skin Thickness"),
    gr.Slider(0, 900, label="Insulin"),
    gr.Slider(0, 130, label="Blood Pressure"),
    gr.Slider(0.0, 3.0, label="Diabetes Pedigree Function")
]

gr.Interface(
    fn=predict_diabetes,
    inputs=inputs,
    outputs="label",
    title="🔍 Diabetes Risk Predictor",
    description="Enter patient health metrics to assess diabetes risk (75% accurate)"
).launch()
