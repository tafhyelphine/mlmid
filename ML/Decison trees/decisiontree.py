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

# Get cost complexity pruning path
path = dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# Try multiple alphas and find the best one
models = []
for alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
    clf.fit(X_train, y_train)
    models.append(clf)

# Evaluate accuracy for each alpha
acc_scores = [accuracy_score(y_test, model.predict(X_test)) for model in models]

# Plot alpha vs accuracy
plt.figure(figsize=(8, 5))
plt.plot(ccp_alphas, acc_scores, marker='o')
plt.xlabel("ccp_alpha (Post-pruning strength)")
plt.ylabel("Accuracy on Test Set")
plt.title("Post-Pruning: Alpha vs Accuracy")
plt.grid(True)
plt.show()

# Pick the best alpha
best_index = np.argmax(acc_scores)
best_alpha = ccp_alphas[best_index]
print(f"Best alpha for post-pruning: {best_alpha:.4f}")

# Final post-pruned tree
dt_postprune = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha)
dt_postprune.fit(X_train, y_train)
y_pred_post = dt_postprune.predict(X_test)
print(f"Post-pruned Tree Accuracy: {accuracy_score(y_test, y_pred_post):.4f}")
