import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os

# Load dataset
print("Loading dataset...")
df = pd.read_csv("data/text_emotion.csv")

print("\nDataset Loaded ✅")
print("Shape:", df.shape)
print("Columns:", df.columns)
print(df.head())

# The dataset has text + emotion in one column (e.g. "I'm happy;joy")
# So split it into two parts
df[['text', 'emotion']] = df['content'].str.split(';', n=1, expand=True)

# Drop rows where either text or emotion is missing
df = df.dropna(subset=['text', 'emotion'])

# Strip spaces from both columns
df['text'] = df['text'].str.strip()
df['emotion'] = df['emotion'].str.strip()

print("\n✅ Extracted columns successfully:")
print(df.head())

# Define features and labels
X = df['text']
y = df['emotion']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text data
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# Evaluate
score = model.score(X_test_vec, y_test)
print(f"\n✅ Model trained successfully! Accuracy: {score*100:.2f}%")

# Create models directory if not exists
os.makedirs("models", exist_ok=True)

# Save model and vectorizer
with open("models/text_emotion_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\n✅ Model and vectorizer saved to 'models/' folder!")
