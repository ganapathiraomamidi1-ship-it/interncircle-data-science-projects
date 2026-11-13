from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "titanic_decision_tree_pipeline.joblib"

st.set_page_config(page_title="Titanic Survival Explorer", page_icon="🚢", layout="centered")
st.title("Titanic Survival Explorer")
st.caption("A small educational demo using a decision tree trained on the public Titanic sample.")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.info("Enter a sample passenger profile. The model returns a historical pattern estimate, not a causal explanation or real-world advice.")

left, right = st.columns(2)
with left:
    pclass = st.selectbox("Passenger class", options=[1, 2, 3], index=0, format_func=lambda value: {1: "First", 2: "Second", 3: "Third"}[value])
    sex = st.selectbox("Sex recorded in dataset", options=["female", "male"], index=1)
    age = st.slider("Age", min_value=0.0, max_value=80.0, value=30.0, step=1.0)
    fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=40.0, step=5.0)
with right:
    sibsp = st.number_input("Siblings / spouses aboard", min_value=0, max_value=8, value=0, step=1)
    parch = st.number_input("Parents / children aboard", min_value=0, max_value=6, value=0, step=1)
    embarked = st.selectbox("Port recorded", options=["S", "C", "Q"], format_func=lambda value: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[value])

if st.button("Predict", type="primary"):
    sample = pd.DataFrame([{
        "pclass": pclass, "sex": sex, "age": age, "sibsp": sibsp,
        "parch": parch, "fare": fare, "embarked": embarked,
    }])
    model = load_model()
    probability = float(model.predict_proba(sample)[0, 1])
    predicted = int(model.predict(sample)[0])
    st.metric("Estimated probability of recorded survival", f"{probability:.1%}")
    st.progress(probability)
    st.write("Model label:", "**Survived**" if predicted else "**Did not survive**")
    st.caption("This is a model output on a historical dataset; it does not explain causes or predict outcomes for real people.")
