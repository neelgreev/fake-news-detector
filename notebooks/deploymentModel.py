import streamlit as st
import pickle

# Load model and vectorizer using pickle
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detector (Manual with Pickle)")
st.markdown("Paste any news article or headline and get a prediction:")

# Input from user
user_input = st.text_area("Enter News Text:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text first.")
    else:
        # Transform input using loaded vectorizer
        X_input = vectorizer.transform([user_input])
        prediction = model.predict(X_input)[0]
        probability = model.predict_proba(X_input)[0]
        confidence = max(probability)

        # Show result
        if prediction == 1:
            st.success(f"✅ The news is **REAL** with {confidence:.2%} confidence.")
        else:
            st.error(f"🚨 The news is **FAKE** with {confidence:.2%} confidence.")