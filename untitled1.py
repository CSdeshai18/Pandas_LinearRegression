import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('GlobalEmission.csv')

dummies = pd.get_dummies(df, dtype='int64')  # Removed cumulative_luc_co2 reference
print(dummies.head())

# Drop 'co2' column if it exists
if 'co2' in dummies.columns:
    data1 = dummies.drop(columns=['co2'])
else:
    data1 = dummies
print(data1.head())
print(data1.columns)

# Define features and target
features = ['population', 'year', 'land_use_change_co2', 'land_use_change_co2_per_capita']
target = 'co2'

# Handle missing values
df = df.fillna(0)

# Define X (features) and y (target)
X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Standardize the features for kNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- LINEAR REGRESSION ---- #
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_lr = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, y_pred_lr)
lr_r2 = r2_score(y_test, y_pred_lr)

print("\nLinear Regression Results:")
print("Coefficients:", lr_model.coef_)
print("Intercept:", lr_model.intercept_)
print("Mean Squared Error:", lr_mse)
print("R² Score:", lr_r2)

# ---- k-NEAREST NEIGHBORS ---- #
# Initialize kNN regressor
knn_model = KNeighborsRegressor(n_neighbors=5)  # Default k=5
knn_model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred_knn = knn_model.predict(X_test_scaled)
knn_mse = mean_squared_error(y_test, y_pred_knn)
knn_r2 = r2_score(y_test, y_pred_knn)

print("\nk-Nearest Neighbors Results:")
print("Mean Squared Error:", knn_mse)
print("R² Score:", knn_r2)

# ---- OPTIONAL: Finding Optimal k ---- #
errors = []
for k in range(1, 21):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    errors.append(mean_squared_error(y_test, y_pred))

# Plot k vs. MSE
plt.figure(figsize=(8, 6))
plt.plot(range(1, 21), errors, marker='o')
plt.title("k vs. Mean Squared Error")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Mean Squared Error")
plt.show()

# Scatter Plot
sns.scatterplot(data=df, x='population', y='year')
plt.show()


