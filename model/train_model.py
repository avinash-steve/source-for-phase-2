import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Sample data: you can replace this with your own dataset
data = {
    'text': ['I am happy', 'I am sad', 'I am angry', 'I am excited', 'I am frustrated'],
    'emotion': ['happy', 'sad', 'angry', 'happy', 'angry']
}
df = pd.DataFrame(data)

# Features (text) and labels (emotions)
X = df['text']
y = df['emotion']

# Convert text to numbers
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train the model
model = LogisticRegression()
model.fit(X_vectorized, y)

# Save the model
with open('emotion_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the vectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved successfully.")
