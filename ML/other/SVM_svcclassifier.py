import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load dataset with correct delimiter
df = pd.read_csv(r"C:\Users\srihari\Desktop\ML\datasets\cardio.csv", delimiter=";")

# Reduce dataset size if too slow
df = df.sample(5000, random_state=42)  # Take a random sample of 5000 rows

# Select features and target
X = df[['age', 'height', 'weight', 'ap_hi', 'ap_lo']]
y = df['cardio']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM with RBF kernel
model = SVC(kernel="rbf", C=1.0, gamma='scale')  # Try reducing C if it's still slow
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Classifier Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
