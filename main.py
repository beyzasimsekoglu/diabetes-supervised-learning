import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [12, 4]

# Load the diabetes dataset
diabetes = datasets.load_diabetes(as_frame=True)
diabetes_X, diabetes_y = diabetes.data, diabetes.target

# Print part of the dataset
print(diabetes_X.head())

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

y1 = np.array([1, 2, 3, 4])   # True values
y2 = np.array([-1, 1, 3, 5])  # Predicted values

print('Mean squared error: %.2f' % mean_squared_error(y1, y2))
print('Mean absolute error: %.2f' % mean_absolute_error(y1, y2))

# . In scikit-learn’s LinearRegression
# The optimizer is built-in.
# It automatically finds the best θ (weights)
# to minimize the mean squared error (MSE) on the training data.

diabetes_X_train = diabetes_X.iloc[-20:]
diabetes_y_train = diabetes_y.iloc[-20:]

# creates a linear regression model / Trains it on the training data
# (optimizer finds the best 0 to minimize MSE

from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train.values)

# Predicts diabetes risk for the training patients.
diabetes_y_train_pred = regr.predict(diabetes_X_train)

#Selects the first 3 patients for testing (new, unseen data).
diabetes_X_test = diabetes_X.iloc[:3]
diabetes_y_test = diabetes_y.iloc[:3]

#Predicts diabetes risk for the new patients.
diabetes_y_test_pred = regr.predict(diabetes_X_test)

# visualize the result
plt.xlabel('Body Mass Index (BMI)')
plt.ylabel('Diabetes Risk')
plt.scatter(diabetes_X_train.loc[:, ['bmi']], diabetes_y_train)
plt.scatter(diabetes_X_test.loc[:, ['bmi']], diabetes_y_test, color='red', marker='o')
plt.plot(diabetes_X_test.loc[:, ['bmi']], diabetes_y_test_pred, 'x', color='red', mew=3, markersize=8)
plt.legend(['Model', 'Prediction', 'Initial patients', 'New patients'])
plt.show()

#. What’s Happening Statistically?
#The optimizer (in LinearRegression) finds the line that minimizes the MSE on the training data.
#You can measure the quality of the fit by comparing predictions to actual values (using MSE, MAE, etc.).
#Visualizing helps you see how well the model generalizes to new data.


from sklearn.metrics import mean_squared_error

print('Training set mean squared error: %.2f'
      % mean_squared_error(diabetes_y_train, diabetes_y_train_pred))
print('Test set mean squared error: %.2f'
      % mean_squared_error(diabetes_y_test, diabetes_y_test_pred))
print('Test set mean squared error on random inputs: %.2f'
      % mean_squared_error(diabetes_y_test, np.random.randn(*diabetes_y_test_pred.shape)))

## A model class: the set of possible models we consider.
#An objective function, which defines how good a model is.
#An optimizer, which finds the best predictive model in
# the model class according to the objective function


## Model Class: The set of possible models

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load your diabetes dataset
diabetes = datasets.load_diabetes(as_frame=True)
diabetes_X, diabetes_y = diabetes.data, diabetes.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    diabetes_X, diabetes_y, test_size=0.2, random_state=42
)

print("=== MODEL CLASSES ===")
print("We can consider different types of models:")

# 1. Linear Regression (My current model)
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_score = linear_model.score(X_test, y_test)
print(f"1. Linear Regression R²: {linear_score:.4f}")

# 2. Ridge Regression (linear with regularization)
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
ridge_score = ridge_model.score(X_test, y_test)
print(f"2. Ridge Regression R²: {ridge_score:.4f}")

# 3. Decision Tree (non-linear model)
tree_model = DecisionTreeRegressor(max_depth=5)
tree_model.fit(X_train, y_train)
tree_score = tree_model.score(X_test, y_test)
print(f"3. Decision Tree R²: {tree_score:.4f}")

# 4. Random Forest (ensemble model)
forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
forest_model.fit(X_train, y_train)
forest_score = forest_model.score(X_test, y_test)
print(f"4. Random Forest R²: {forest_score:.4f}")

print("\n=== OPTIMIZERS ===")
print("Each model uses different optimization algorithms:")

# Linear Regression optimizer (Ordinary Least Squares)
print("1. Linear Regression:")
print(f"   - Uses: Ordinary Least Squares (OLS)")
print(f"   - Finds: Best coefficients to minimize MSE")
print(f"   - Parameters found: {len(linear_model.coef_)} coefficients + 1 intercept")

# Ridge Regression optimizer (with regularization)
print("\n2. Ridge Regression:")
print(f"   - Uses: Ridge regression with L2 penalty")
print(f"   - Finds: Best coefficients while keeping them small")
print(f"   - Alpha (regularization strength): {ridge_model.alpha}")

# Decision Tree optimizer (greedy algorithm)
print("\n3. Decision Tree:")
print(f"   - Uses: Greedy algorithm to find best splits")
print(f"   - Finds: Best feature and threshold at each node")
print(f"   - Max depth: {tree_model.max_depth}")

# Random Forest optimizer (ensemble method)
print("\n4. Random Forest:")
print(f"   - Uses: Bootstrap aggregating (bagging)")
print(f"   - Finds: Best combination of multiple trees")
print(f"   - Number of trees: {forest_model.n_estimators}")

# Compare all models
models = {
    'Linear': linear_model,
    'Ridge': ridge_model,
    'Tree': tree_model,
    'Forest': forest_model
}

scores = []
names = []

for name, model in models.items():
    score = model.score(X_test, y_test)
    scores.append(score)
    names.append(name)

# Plot comparison
plt.figure(figsize=(10, 6))
plt.bar(names, scores, color=['blue', 'green', 'orange', 'red'])
plt.ylabel('R² Score')
plt.title('Model Performance Comparison')
plt.ylim(0, 1)
for i, v in enumerate(scores):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
plt.show()

print(f"\nBest performing model: {names[scores.index(max(scores))]} (R² = {max(scores):.4f})")


#| Model | Best For | Optimizer | Why That Optimizer |
#|-------|----------|-----------|-------------------|
#| Linear Regression | Linear relationships, interpretable results | OLS | Fast, exact, mathematically optimal |
#| Ridge Regression | High-dimensional data, correlated features | L2 regularization | Prevents overfitting, handles multicollinearity |
#| Decision Trees | Non-linear relationships, categorical features | Greedy algorithm | Simple, fast, captures complex patterns |
#| Random Forest | Complex data, noisy data, robust predictions | Bagging | Reduces overfitting, averages multiple models |
