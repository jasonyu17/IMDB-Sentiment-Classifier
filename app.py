import streamlit as st
import joblib
import os
import json
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json


# --- Helper / config ---
SVM_PATH = "models/svm_model_rbf.joblib"
TFIDF_PATH = "models/tfidf_vectorizer.joblib"
KERAS_MODEL_PATH = "models/nn_models/best.keras"
TOKENIZER_PATH = "models/tokenizer.json"
MAX_LENGTH = 300


def basic_clean(text: str) -> str:
    # lightweight cleaning similar to training-time preprocessing (best-effort)
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --- Load classical model (SVM + TFIDF) if available ---
svm_model = None
tfidf = None
if os.path.exists(SVM_PATH) and os.path.exists(TFIDF_PATH):
    try:
        svm_model = joblib.load(SVM_PATH)
        tfidf = joblib.load(TFIDF_PATH)
    except Exception as e:
        st.warning(f"Could not load SVM/TF-IDF: {e}")


# --- Load Keras model + tokenizer if available ---
keras_model = None
tokenizer = None
if os.path.exists(KERAS_MODEL_PATH):
    try:
        keras_model = tf.keras.models.load_model(KERAS_MODEL_PATH)
    except Exception as e:
        st.warning(f"Could not load Keras model: {e}")

if os.path.exists(TOKENIZER_PATH):
    try:
        with open(TOKENIZER_PATH, "r") as fh:
            tok_json = fh.read()
        tokenizer = tokenizer_from_json(tok_json)
    except Exception as e:
        st.warning(f"Could not load tokenizer: {e}")


# --- Streamlit UI ---
st.title("IMDB Sentiment Classifier")
st.markdown("Enter a movie review and choose a model to predict whether it's **positive** or **negative**.")

available_models = []
if svm_model is not None and tfidf is not None:
    available_models.append("SVM (TF-IDF)")
if keras_model is not None and tokenizer is not None:
    available_models.append("Keras NN")

if not available_models:
    st.error("No models available. Place trained models in the `models/` folder (SVM or Keras).")
else:
    model_choice = st.radio("Model", available_models)

    review = st.text_area("Your Review", height=200)

    if st.button("Predict"):
        if review.strip() == "":
            st.warning("Please enter a review.")
        else:
            if model_choice == "SVM (TF-IDF)":
                # classical pipeline
                X_input = tfidf.transform([review])
                prediction = svm_model.predict(X_input)[0]

                if hasattr(svm_model, "predict_proba"):
                    prob = svm_model.predict_proba(X_input)[0]
                    confidence = max(prob)
                else:
                    confidence = None

                label = "Positive" if prediction == 1 else "Negative"
                st.success(f"**Prediction:** {label}")
                if confidence is not None:
                    st.info(f"Confidence: {confidence:.2%}")

            elif model_choice == "Keras NN":
                # try to use same preprocessing used during training if available
                cleaned = basic_clean(review)

                # tokenize + pad
                seq = tokenizer.texts_to_sequences([cleaned])
                pad = pad_sequences(seq, maxlen=MAX_LENGTH, padding='post', truncating='post')

                # predict probability (assumes final activation is sigmoid)
                try:
                    p = float(keras_model.predict(pad)[0][0])
                except Exception as e:
                    st.error(f"Model prediction failed: {e}")
                    p = None

                if p is not None:
                    label = "Positive" if p >= 0.5 else "Negative"
                    st.success(f"**Prediction:** {label}")
                    st.info(f"Confidence: {p:.2%}")
import streamlit as st
import streamlit as st
import joblib

# Load the saved model and vectorizer
model = joblib.load("models/svm_model_rbf.joblib")
tfidf = joblib.load("models/tfidf_vectorizer.joblib")
  


st.title("IMDB Sentiment Classifier (SVM with RBF Kernel)")
st.markdown("Enter a movie review and the model will predict if it's **positive** or **negative**.")

# Text input
review = st.text_area("Your Review", height=200)

if st.button("Predict"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
 
        X_input = tfidf.transform([review])
        prediction = model.predict(X_input)[0]


        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_input)[0]
            confidence = max(prob)
        else:
            confidence = None

        label = "Positive" if prediction == 1 else "Negative"
        st.success(f"**Prediction:** {label}")
        if confidence is not None:
            st.info(f"Confidence: {confidence:.2%}")
