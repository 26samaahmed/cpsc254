import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


X1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
X1 = np.array(X1).reshape(-1, 1) # Reshaping X to be a 2D array
Y1 = [2.5, 4.1, 5.6, 7.2, 8.8, 10.3, 11.9, 13.5, 15.0, 16.8]
X2 = [-3, -2.5, -2, -1.5, -1, 0, 1, 1.5, 2, 2.5, 3]
X2 = np.array(X2).reshape(-1, 1)
Y2 = [17.5, 12.9, 9.5, 7.2, 5.8, 5.5, 7.1, 9.7, 13.5, 18.4, 24.4]


def plot_data(X, Y):
  plt.scatter(X, Y)
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.title('Scatter plot of Data Set 1')
  plt.show()

def predict_linear_value(X, Y, value):
  model = LinearRegression().fit(X, Y)
  return model.predict([[value]])[0] # Returning only the first predicted value

def predict_polynomial_value(X, Y, value):
  poly = PolynomialFeatures(degree=2)
  X_poly = poly.fit_transform(X)
  model = LinearRegression().fit(X_poly, Y)
  return model.predict(poly.transform([[value]]))[0]

# Display List of Coeficients
def display_coef(X, Y):
  poly = PolynomialFeatures(degree=2) # Change degree to 2 for quadratic regression
  X_poly = poly.fit_transform(X) # Transforming X to polynomial features
  model = LinearRegression().fit(X_poly, Y)
  return model.coef_

def compute_total_error(X, Y):
  model = LinearRegression().fit(X, Y)
  Y_pred = model.predict(X)
  return np.sum((Y - Y_pred))


def compute_sum_error_squared(X, Y):
  model = LinearRegression().fit(X, Y)
  Y_pred = model.predict(X)
  return np.sum((Y - Y_pred) ** 2)

def compute_mean_square_error(X, Y):
  res = compute_sum_error_squared(X, Y)
  return res / len(X) 

def compute_root_mean_square_error(X, Y):
  res = compute_mean_square_error(X, Y)
  return np.sqrt(res)

if __name__ == "__main__":
    print("Predicted value of Y for X = 100: ", predict_linear_value(X1, Y1, 100))
    print("Coefficients of the linear regression model for X2 and Y2: ", display_coef(X2, Y2))
    print("Predicted value of Y for X = 0.5: ", predict_polynomial_value(X2, Y2))

    print("Total Error for X1 and Y1: ", compute_total_error(X1, Y1))
    print("Sum of Squared Errors for X1 and Y1: ", compute_sum_error_squared(X1, Y1))
    print("Mean Square Error for X1 and Y1: ", compute_mean_square_error(X1, Y1))
    print("Root Mean Square Error for X1 and Y1: ", compute_root_mean_square_error(X1, Y1))
