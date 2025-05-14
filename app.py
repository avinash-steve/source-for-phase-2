import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already present
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
with open("emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Preprocess function
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Streamlit UI
st.title("😊 Emotion Detection from Text")

st.markdown("Enter a sentence to predict the emotion:")

user_input = st.text_area("Type your text here")

if st.button("Predict Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned_text = preprocess(user_input)
        vect_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vect_text)[0]
        st.success(f"**Predicted Emotion:** {prediction}")
