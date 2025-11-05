# Import libraries
import streamlit as st
import pandas as pd
import joblib

# Set Page Configuration
st.set_page_config(
    page_title="Burn Rate Prediction",
    page_icon="🔥",
    initial_sidebar_state="expanded"
)

# Add background image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://cdn.pixabay.com/photo/2016/12/29/18/44/background-1939128_1280.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the saved model, and scaler
model = joblib.load("linear_regressor.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

#App title
st.title("Employee Burn Rate Prediction")
st.write("Enter the Working Conditions to Predict Burnout Rate")

#Input fields for the burnout rate prediction
gender = st.selectbox("Gender of the Employee",
                      ["Male", "Female"])
company_type = st.selectbox("The type of company (Either Product or Services)",
                            ["Product", "Service"])
wfh = st.selectbox("Whether or the Employee has option of working from home",
                   ["Yes", "No"])
designation = st.number_input("Designation (Role and Seniority Level of Employee)",
                              min_value=0, max_value=5, step=1
                              )
work_hours = st.number_input("Work Hours (Number of hour the Employee Works)",
                                      min_value=0, max_value=10, step=1
                                    )
fatigue_score = st.slider("Fatigue Score",
                                 min_value=0.0, max_value=10.0, step=0.1, format="%.1f"
                                 )

#Predict the burn rate
if st.button("Predict Burn Rate"):

    
    #Input data
    input_data = pd.DataFrame({
        "gender": [gender],
        "company_type": [company_type],
        "wfh": [wfh],
        "designation": [designation],
        "work_hours": [work_hours],
        "fatigue_score": [fatigue_score]
    })

    #Separate categorical and numerical features
    input_cat = input_data[["gender", "company_type", "wfh"]]
    input_num = input_data.drop(columns=["gender", "company_type", "wfh"])

    #Applying transformation on numerical columns
    input_num_trans = pd.DataFrame(
        scaler.transform(input_num),
        columns=scaler.get_feature_names_out(),
        index=input_num.index
    )

    #Applying transformation on categorical columns
    input_cat_trans = pd.DataFrame(
        encoder.transform(input_cat),
        columns=encoder.get_feature_names_out(["gender", "company_type", "wfh"]),
        index=input_cat.index
    )

    #Concatinating numerical and categorical data
    data = pd.concat([input_num_trans, input_cat_trans], axis=1)
    data = data[model.feature_names_in_]

    #Predict using loaded model
    prediction = model.predict(data)

    #Display the prediction
    st.success(f"The Predicted Burn Rate for this Employee is: {prediction[0]:,.2f}")