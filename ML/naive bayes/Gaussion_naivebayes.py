import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load the dataset
df = pd.read_csv(r"C:\Users\srihari\Desktop\ML\datasets\abalone.csv")  # Replace with actual filename if needed

# Step 2: Select Features (Height, Length) and Target (Sex)
X = df[['Height', 'Length']]
y = df['Sex']  # M, F, or I

# Step 3: Convert Categorical Labels to Numerical
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Converts 'M', 'F', 'I' -> 0, 1, 2

# Step 4: Split Data into Training & Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train Naïve Bayes Model
model = GaussianNB()
model.fit(X_train, y_train)

# Step 6: Make Predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 8: Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Step 9: Visualization (Scatter Plot of Predictions)
plt.scatter(X_test['Length'], X_test['Height'], c=y_pred, cmap='viridis', alpha=0.7)
plt.xlabel("Length")
plt.ylabel("Height")
plt.title("Naïve Bayes Classification of Snails (M/F/I)")
plt.colorbar(label="Predicted Category")
plt.show()
