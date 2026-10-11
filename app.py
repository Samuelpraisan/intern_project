
import streamlit as st
import numpy as np
import joblib

# Load saved model
saved_data = joblib.load("iris_model.pkl")
model = saved_data["model"]
target_names = saved_data["target_names"]

# Page configuration
st.set_page_config(page_title="Iris Predictor", page_icon="🌸")

st.title("🌸 Iris Flower Prediction")
st.write("Adjust the flower measurements to predict its species.")

# Input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5, 0.1)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2, 0.1)

# Prepare input
input_data = np.array([[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]])

# Predict
if st.button("Predict Flower"):
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    st.success(f"Predicted Species: {target_names[prediction]}")
    st.write(f"Prediction confidence: {probabilities[prediction]:.2%}")

    st.subheader("Class probabilities")
    for i, name in enumerate(target_names):
        st.write(f"{name}: {probabilities[i]:.2%}")
