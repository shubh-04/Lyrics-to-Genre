import streamlit as st
import joblib
import re
import numpy as np
from langdetect import detect, DetectorFactory

# make language detection deterministic
DetectorFactory.seed = 0

# -----------------------
# Text cleaning
# -----------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# -----------------------
# Load models lazily (FAST)
# -----------------------
@st.cache_resource
def load_language_model(lang):
    return {
        "model": joblib.load(f"model_{lang}.pkl"),
        "vectorizer": joblib.load(f"tfidf_{lang}.pkl"),
        "encoder": joblib.load(f"label_encoder_{lang}.pkl")
    }

SUPPORTED_LANGUAGES = ["en", "pt"]

# -----------------------
# Language detection
# -----------------------
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

# -----------------------
# UI
# -----------------------
st.set_page_config(
    page_title="Lyrics Genre Classifier",
    page_icon="🎵",
    layout="centered"
)

st.title("🎵 Multilingual Lyrics Genre Classifier")
st.write(
    "Paste song lyrics below. "
    "The system automatically detects the language and predicts the genre."
)

lyrics = st.text_area(
    "🎤 Song Lyrics",
    height=260,
    placeholder="Paste at least a few lines of lyrics here..."
)

if st.button("🚀 Predict Genre"):
    if lyrics.strip() == "":
        st.warning("Please enter some lyrics.")
        st.stop()

    detected_lang = detect_language(lyrics)
    st.write(f"🌍 Detected Language: **{detected_lang}**")

    if detected_lang not in SUPPORTED_LANGUAGES:
        st.error("This language is not supported yet.")
        st.stop()

    # ---- CLEAN TEXT ----
    cleaned = clean_text(lyrics)

    # ---- GUARDRAIL FOR SHORT INPUT ----
    if len(cleaned.split()) < 20:
        st.warning("Please enter at least **20 words** for an accurate prediction.")
        st.stop()

    # ---- LOAD MODEL FOR THIS LANGUAGE ----
    assets = load_language_model(detected_lang)

    # ---- VECTORIZE ----
    X_text = assets["vectorizer"].transform([cleaned])

    # ---- PREDICT PROBABILITIES ----
    probs = assets["model"].predict_proba(X_text)[0]
    classes = assets["encoder"].classes_

    # ---- TOP 3 ----
    top_idx = np.argsort(probs)[::-1][:3]

    st.subheader("🎧 Top Genre Predictions")

    for rank, i in enumerate(top_idx, start=1):
        genre = classes[i]
        confidence = probs[i]

        st.markdown(f"**{rank}. {genre}**")
        st.progress(float(confidence))
        st.caption(f"{confidence * 100:.2f}% confidence")
