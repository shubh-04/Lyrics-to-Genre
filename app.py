import streamlit as st
import joblib
import re
from langdetect import detect, DetectorFactory

# Make language detection deterministic
DetectorFactory.seed = 0

# -----------------------
# Text cleaning function
# -----------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# -----------------------
# Load models (cached)
# -----------------------
@st.cache_resource
def load_models():
    models = {
        "en": {
            "model": joblib.load("model_en.pkl"),
            "vectorizer": joblib.load("tfidf_en.pkl"),
            "encoder": joblib.load("label_encoder_en.pkl")
        },
        "pt": {
            "model": joblib.load("model_pt.pkl"),
            "vectorizer": joblib.load("tfidf_pt.pkl"),
            "encoder": joblib.load("label_encoder_pt.pkl")
        }
    }
    return models

models = load_models()

# -----------------------
# Language detection
# -----------------------
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Lyrics Genre Classifier", page_icon="🎵")

st.title("🎵 Multilingual Lyrics Genre Classifier")
st.write("Paste song lyrics below. Language will be detected automatically.")

lyrics = st.text_area("Song Lyrics", height=250)

if st.button("Predict Genre"):
    if lyrics.strip() == "":
        st.warning("Please enter some lyrics.")
    else:
        detected_lang = detect_language(lyrics)
        st.write(f"Detected Language: **{detected_lang}**")

        if detected_lang not in models:
            st.error("This language is not supported yet.")
        else:
            cleaned = clean_text(lyrics)
            vector = models[detected_lang]["vectorizer"].transform([cleaned])
            pred = models[detected_lang]["model"].predict(vector)[0]
            genre = models[detected_lang]["encoder"].inverse_transform([pred])[0]

            st.success(f"🎶 Predicted Genre: **{genre}**")
