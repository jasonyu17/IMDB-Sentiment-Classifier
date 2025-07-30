import streamlit as st
import joblib

# Load the saved model and vectorizer
model = joblib.load("models/svm_model_rbf.joblib")
tfidf = joblib.load("models/tfidf_vectorizer.joblib")
  


st.title("IMDB Sentiment Classifier (Logistic Regression)")
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
