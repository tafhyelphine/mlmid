import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv(r"C:\Users\srihari\Desktop\ML\datasets\cardio.csv", delimiter=";")

# Convert age from days to years
df["age"] = df["age"] // 365

# Features & Target
X = df.drop(columns=["id", "cardio"])
y = df["cardio"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Model
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)

# Predictions & Accuracy
y_pred_dt = dt.predict(X_test)
print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")

# Decision Tree Plot
plt.figure(figsize=(12, 6))
plot_tree(dt, feature_names=X.columns, class_names=["No Disease", "Has Disease"], filled=True)
plt.title("Decision Tree Visualization")
plt.show()
