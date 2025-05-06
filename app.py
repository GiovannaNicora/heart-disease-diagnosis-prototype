import streamlit as st
import pickle
import os
import pandas as pd
import numpy as np
import shap

def model_proba(x, model):
    return model.predict_proba(x)[:, 1]

def model_log_odds(x, model):
    p = model.predict_log_proba(x)
    return p[:, 1] - p[:, 0]

def predict(model, data):
    y_pred = model.predict(data)
    y_pred_proba = model.predict_proba(data)[:, 1]

    return y_pred, y_pred_proba

def run(model, explainer):
    st.sidebar.markdown("Insert Patient's clinical data")
    placeholder = "--select--"

    # mapping
    gender_map = {'Male': 0, 'Female': 1}
    chest_pain_map = {"1-typical angina":1, "2-atypical angina":2,
                      "3-non anginal pain": 3, "4-asymptomatic":4}
    boolean_map = {"Yes":1, "No":0}
    resting_ecg_map = dict(zip(["0-Normal", "1: having ST-T wave abnormality",
                               "2:showing probable or definite left ventricular hypertrophy by Estes' criteria"],
                               [0,1,2]))
    slope_map = dict(zip(['1: upsloping','2: flat','3: downsloping'], [1, 2, 3]))
    thal_map = dict(zip(["3-normal", "6-fixed defect", "7-reversable defect"], [3, 6, 7]))

    # creating input fields
    age = st.sidebar.number_input("Age", min_value=0, max_value=110, value=None, placeholder="Type age")
    sex = st.sidebar.selectbox("Biological sex", options=list(gender_map.keys()),
                                   placeholder="--select--", index=None)
    chest_pain = st.sidebar.selectbox("Type of chest pain",
                                      options=list(chest_pain_map.keys()),
                                      index=None, placeholder="--select--")
    resting_blood_p = st.sidebar.number_input("Resting blood pressure (in mm Hg on admission to the hospital)",
                                              min_value=0, value=None, placeholder="Type resting blood pressure")
    chol = st.sidebar.number_input("serum cholestoral in mg/dl", min_value=0, value=None, placeholder="Type cholesterol")
    fbs = st.sidebar.selectbox("fasting blood sugar > 120 mg/dl", options=list(boolean_map.keys()),
                               placeholder="--select--", index=None)

    resting_ecg = st.sidebar.selectbox("Resting ECG Results", options=list(resting_ecg_map.keys()))
    thalac = st.sidebar.number_input("maximum heart rate achieved",value=None, placeholder="Type max heart rate achieved")
    exang = st.sidebar.selectbox("exercise induced angina", options=list(boolean_map.keys()), index=None,
                                 placeholder="--select--")
    oldpeak = st.sidebar.number_input("ST depression induced by exercise relative to rest", value=None, placeholder="Type ST depression")
    slope = st.sidebar.selectbox("T depression induced by exercise relative to rest",
                                   options=list(slope_map.keys()),
                                   index=None, placeholder="--select--")

    ca = st.sidebar.selectbox("number of major vessels (0-3) colored by flourosopyy",
                              options=['1', '2', '3'], index=None, placeholder="--select--")

    thal = st.sidebar.selectbox("Thalassemia", options=list(thal_map.keys()),
                                index=None, placeholder=placeholder)


    if st.sidebar.button("Predict!"):
        try:
            age = float(age)
            sex = gender_map[sex]
            chest_pain = chest_pain_map[chest_pain]
            resting_blood_p = float(resting_blood_p)
            chol = float(chol)
            fbs = boolean_map[fbs]
            resting_ecg = resting_ecg_map[resting_ecg]
            thalac = float(thalac)
            exang = boolean_map[exang]
            oldpeak = float(oldpeak)
            slope = slope_map[slope]
            ca = int(ca)
            thal = thal_map[thal]
            data = np.array([age, sex, chest_pain, resting_blood_p, chol, fbs, resting_ecg, thalac,
                             exang, oldpeak, slope, ca, thal]).reshape(1, -1)
            data = pd.DataFrame(data=data, columns=model.feature_names_in_)
            ypred, ypredproba = predict(model, data)
            print(ypred, ypredproba)
            st.markdown("Heart disease diagnosis prediction:"+str(ypred[0]))
            st.markdown("Predicted probability of risk: "+ str(ypredproba[0]))


        except ValueError:
            st.markdown("You need to insert all the clinical data!")

     #   shap_val = explainer(data)
     #   print("shap", shap_val)
     #   shap.plots.bar(shap_val)




def main():
    st.title("Heart Disease Prediction")

    model = pickle.load(open(os.getcwd()+"/finalized_model_logistic_regression.pkl", "rb"))
    explainer = pickle.load(open(os.getcwd()+"/explainer_logistic_regression.pkl", "rb"))
    st.markdown("Get cardiovascular risk prediction from clinical features")
    run(model, explainer)


    #left_column, right_column = st.columns(2)
    #left_column.header("Data Input")

if __name__ == "__main__":
    main()
