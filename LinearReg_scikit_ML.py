# Linear_Regression.py
# Linear Regression code example with scikit-learn
# Author: Atefeh Joulaei

import numpy as np
from sklearn.linear_model import LinearRegression ## Linear Regression is a supervised learning algorithm.
import matplotlib.pyplot as plt


# 1️. Data preparation with the input data
x = [1, 2, 3, 4, 5]
y = [3, 5, 7, 9, 11]

x = np.array(x).reshape(-1, 1) # scikit-learn expects data to be in 2D column. so, it reshapes the x-array.
y = np.array(y)

# 2. Model creation
model = LinearRegression()
model.fit(x, y) # Fit/Learn the model with input data

# 3️. Predictions
y_pred = model.predict(x)
print("Predicted values for training data:", y_pred)

# prediction of the model with the new Entry
x_new = np.array([[6]])
y_pred_new = model.predict(x_new)
print("Predicted Y for x = 6:", y_pred_new[0])

# 4️. Evaluation: slope, intercept, score and r calculation
slope = model.coef_
intercept = model.intercept_
score = model.score(x, y) # R^2 Score shows how well the model explains the variance of the data (not exactly correlation coefficient)
r = np.corrcoef(x.flatten(), y)[0, 1] # coefficient of correlation

# 5️. Print the Result
print("Slope:", slope[0]) # slope is an array (for multiple features). slope[0] is the coefficient for x.
print("Intercept:", intercept)
print("R^2 Score (model performance):", score) # shows that how does the model explain the data. is it correlation coefficient
print("Correlation coefficient r (relationship strength):", r)


# 6️. Visualization
plt.scatter(x.flatten(), y)
plt.plot(x.flatten(), y_pred)
plt.title("Linear Regression Example")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


