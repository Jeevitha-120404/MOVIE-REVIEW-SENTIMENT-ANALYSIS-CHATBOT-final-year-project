import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from preprocess import clean_text

# Load labeled dataset
df = pd.read_csv("movie_reviews_dataset.csv")

# Clean text
df["clean_review"] = df["Review"].apply(clean_text)

# Features & labels
X = df["clean_review"]
y = df["Sentiment"]   # Positive / Neutral / Negative

# Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Logistic Regression (multiclass)
model = LogisticRegression(
    solver="lbfgs",
    max_iter=1000
)

model.fit(X_vec, y)

import joblib

# Save model and vectorizer
joblib.dump((model, vectorizer), "sentiment_model.pkl")

print("✅ Model and Vectorizer saved successfully")
