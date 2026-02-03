import streamlit as st
from cleanning import clean_text
import joblib
import numpy as np

# Load model and vectorizer
count = joblib.load("count.joblib")
model = joblib.load("model.joblib")

# Page configuration
st.set_page_config(page_title="Finance Sentiment Detector", page_icon="💼")
st.balloons()

# Main title
st.title("💼 Finance Sentiment Analysis")

# Subtitle
st.markdown("Analyze the sentiment of financial news, tweets, or reports using a machine learning model.")

# Input area
text = st.text_area("📝 Enter your financial text here:", height=150)

# Centered button with columns
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    submit = st.button("🔍 Analyze")

# Handle submission
if submit:
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Processing..."):
            cleaned = clean_text(text)
            vector = count.transform([cleaned]).todense()
            prediction = model.predict(vector)[0]

        # Mapping and styling output
        sentiment_map = {
            0: ("Negative", "🔴", "danger"),
            1: ("Neutral", "🟡", "warning"),
            2: ("Positive", "🟢", "success")
        }

        label, emoji, tag = sentiment_map.get(prediction, ("Unknown", "❓", "info"))

        st.markdown(f"### {emoji} Sentiment: **{label}**")

        # Optional: display raw prediction value
        with st.expander("See raw model output"):
            st.code(f"Predicted label: {prediction}", language="text")

# Optional footer
st.markdown("""---  
Made with ❤️ using Streamlit | [GitHub](https://github.com/)  
""")
