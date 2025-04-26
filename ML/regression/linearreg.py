import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load dataset
house_df = pd.read_csv(r"C:\Users\tafhy\Downloads\BostonHousing (1).csv")  # Update your path if needed

# EDA
print("\n--- Data Info ---")
print(house_df.info())

print("\n--- Summary Statistics ---")
print(house_df.describe())

print("\n--- Missing Values ---")
print(house_df.isna().sum())

house_df['rm'].fillna(house_df['rm'].mean(), inplace = True)
print(house_df.isna().sum())

# Outlier removal using Z-score
numeric_cols = house_df.select_dtypes(include=[np.number]).columns
z_scores = np.abs((house_df[numeric_cols] - house_df[numeric_cols].mean()) / house_df[numeric_cols].std())
house_df_clean = house_df[(z_scores < 3).all(axis=1)]

print(f"\nOriginal shape: {house_df.shape}")
print(f"Shape after removing outliers: {house_df_clean.shape}")

# Feature scaling
features = house_df_clean.drop('medv', axis=1)
target = house_df_clean['medv']

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

scaled_df = pd.DataFrame(scaled_features, columns=features.columns)

# Simple Linear Regression
X_train, X_test, y_train, y_test = train_test_split(scaled_df, target, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Performance ---")
print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'R-squared: {r2:.2f}')

# Plot: Actual vs Predicted
plt.scatter(y_test, y_pred)
plt.xlabel('Actual MEDV')
plt.ylabel('Predicted MEDV')
plt.title('Actual vs Predicted MEDV')
plt.show()

# Residual plot
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted MEDV')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')
plt.show()

for feature in features.columns:
    plt.figure(figsize=(8, 5))
    sns.regplot(x=house_df_clean[feature], y=house_df_clean['medv'], line_kws={"color": "red"})
    plt.title(f'Regression Line: {feature} vs MEDV')
    plt.xlabel(feature)
    plt.ylabel('MEDV')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

from scipy.stats import ttest_1samp

# T-test: Is average MEDV significantly different from $20K?
t_stat, p_val = ttest_1samp(house_df_clean['medv'], 20)

print(f"T-statistic = {t_stat:.2f}, P-value = {p_val:.4f}")
print("Significant difference!" if p_val < 0.05 else "No significant difference.")

sns.pairplot(house_df_clean[['medv', 'rm', 'lstat', 'ptratio', 'nox', 'crim']])
plt.suptitle("Pair Plot of Selected Features", y=1.02)
plt.show()

from statsmodels.stats.weightstats import ztest

# Z-test: Is the average MEDV significantly different from $20K?
z_stat, p_val = ztest(house_df_clean['medv'], value=20)

print(f"Z-statistic = {z_stat:.2f}, P-value = {p_val:.4f}")
print("Significant difference!" if p_val < 0.05 else "No significant difference.")

