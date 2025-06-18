import streamlit as st
import joblib
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Text cleaning
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
    tokens = text.split()
    return ' '.join([stemmer.stem(word) for word in tokens if word not in stop_words])

# Load model and data
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
df = joblib.load("tweet_data.pkl")

# --- Page Config ---
st.set_page_config(page_title="Tweet Sentiment Analyzer (Beta)", page_icon="💬", layout="centered")

# --- Custom CSS ---
st.markdown("""
    <style>
    body {
        background: linear-gradient(134deg, #6a11cb 0%, #2575fc 100%);
        color: white;
    }
    .stApp {
        background: linear-gradient(134deg, #6a11cb 0%, #2575fc 100%);
        color: white;
    }
    .stButton>button {
        background-color: #ffffff;
        color: #4a00e0;
        border: none;
        padding: 0.6em 2em;
        font-size: 1.1em;
        border-radius: 8px;
        margin-top: 10px;
        font-weight: bold;
    }
    .stNumberInput input {
        font-size: 1.1em;
    }
    .sentiment {
        font-size: 1.5em;
        font-weight: 600;
        padding: 15px;
        border-radius: 10px;
        background-color: #ffffff20;
        border: 2px solid #ffffff50;
        margin-top: 20px;
        color: white;
        text-align: center;
    }
    .stInfo {
        background-color: #ffffff20 !important;
        border-left: 5px solid #ffffff80;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# --- App Title ---
st.title("🔍 Tweet Sentiment Analyzer (Beta)")

st.markdown("""
Enter a tweet index from the dataset and get an AI-powered **Positive 😊** or **Negative 😠** prediction.
""")

# --- Input ---
index = st.number_input("🔢 Enter tweet index", min_value=0, max_value=len(df)-1, step=1)

# --- Analyze Button ---
if st.button("🎯 Analyze Sentiment"):
    text = df.iloc[index]['text']
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    label = "😊 Positive" if pred == 1 else "😠 Negative"
    emoji = "🟢" if pred == 1 else "🔴"

    st.subheader("📄 Original Tweet:")
    st.info(text)

    st.markdown(f"""
    <div class='sentiment'>
        Sentiment: <b>{label}</b> {emoji}
    </div>
    """, unsafe_allow_html=True)
