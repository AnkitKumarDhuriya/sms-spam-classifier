import streamlit as st
import pickle
from utils import transform_text

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📩",
    layout="centered"
)

# ---------------- Load Model ----------------
with open("models/vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------- Title ----------------
st.title("📩 SMS Spam Classifier")
st.caption("Detect whether an SMS is Spam or Ham using Machine Learning.")

st.divider()

# ---------------- Input ----------------
st.subheader("✍️ Enter Your Message")

input_sms = st.text_area(
    label="",
    placeholder="Type or paste your SMS here...",
    height=150
)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict = st.button("🔍 Predict", use_container_width=True)

# ---------------- Prediction ----------------
if predict:

    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        prediction = model.predict(vector_input)[0]

        st.divider()

        if prediction == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Ham Message")