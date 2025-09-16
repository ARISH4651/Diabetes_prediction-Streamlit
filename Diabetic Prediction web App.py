# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 22:13:43 2025

@author: arish
"""

import numpy as np
import pickle
import streamlit as st

# Load saved model
loaded_model = pickle.load(open('D:/ml deploy/trained_model.sav', 'rb'))


# Prediction function
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return " The patient does not have Diabetes"
    else:
        return " The patient is likely to have Diabetes"

# Streamlit App
def main():
    st.set_page_config(page_title="Disease Prediction App", page_icon="🩺", layout="wide")

    st.title("🩺 Disease Prediction Web App")
    st.write("This application predicts the likelihood of **Diabetes**  **Heart Cancer**.")

    # Tabs for different diseases
    (tab1,) = st.tabs(["Diabetes Prediction"])

    # ---------------- Diabetes Tab ----------------
    with tab1:
        st.header("Diabetes Prediction 🩸")

        col1, col2 = st.columns(2)

        with col1:
            Pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, step=1)
            Glucose = st.number_input("Glucose Level", min_value=0, max_value=300, step=1)
            BloodPressure = st.number_input("Blood Pressure Value", min_value=0, max_value=200, step=1)
            SkinThickness = st.number_input("Skin Thickness Value", min_value=0, max_value=100, step=1)

        with col2:
            Insulin = st.number_input("Insulin Level", min_value=0, max_value=900, step=1)
            BMI = st.number_input("BMI Level", min_value=0.0, max_value=70.0, step=0.1)
            DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function Value", min_value=0.0, max_value=3.0, step=0.01)
            Age = st.number_input("Age of the Person", min_value=1, max_value=120, step=1)

        # Prediction button
        if st.button("🔍 Predict Diabetes"):
            diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness,
                                             Insulin, BMI, DiabetesPedigreeFunction, Age])
            if diagnosis:
                st.success(diagnosis)
            else:
                st.error(diagnosis)

        st.image("D:\ml deploy\diabetes-risk_orig.png",
                 caption="Diabetes Risk Factors", use_container_width=True)



# Run main
if __name__ == '__main__':
    main() 
