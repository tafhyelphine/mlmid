# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Load Dataset
data = pd.read_csv(r"C:\Users\srihari\Desktop\ML\datasets\spam_ham_dataset (1).csv")  # Load dataset

# Step 3: Define Features (X) and Target (y)
X = data["text"]  # Text messages
y = data["label"].map({'spam': 1, 'ham': 0})  # Convert labels to numerical (spam=1, ham=0)

# Step 4: Split Data into Train & Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Convert Text to Numerical Features using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)  # Transform training data
X_test_tfidf = vectorizer.transform(X_test)  # Transform test data

# Step 6: Train Multinomial Naïve Bayes Model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Step 7: Make Predictions
y_pred = model.predict(X_test_tfidf)

# Step 8: Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 9: Visualizing Spam vs Ham Count
ham_count = sum(y_test == 0)
spam_count = sum(y_test == 1)

plt.bar(["Ham", "Spam"], [ham_count, spam_count], color=['blue', 'red'])
plt.title("Ham vs Spam Count in Test Data")
plt.ylabel("Count")
plt.show()
