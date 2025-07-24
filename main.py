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