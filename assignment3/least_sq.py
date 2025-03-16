from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import os


csv_path = '../assignment2/Q2.csv'
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Error: The file '{csv_path}' was not found. Please check the path.")

df = pd.read_csv(csv_path)
X = df[['x1', 'x2']].values  
Y = df['y'].values

# Add bias (intercept) column of ones
X_bias = np.column_stack((np.ones(X.shape[0]), X))


def find_beta(X, Y):
    return np.linalg.inv(X.T @ X) @ X.T @ Y

beta = find_beta(X_bias, Y)


equation_terms = []
for i in range(len(beta)):
    if i == 0:
        term = f"{beta[i]:.4f}"
    else:
        term = f"{beta[i]:.4f} * x{i}"
    equation_terms.append(term)
equation = " + ".join(equation_terms)
print("\nPolynomial Equation from Least Squares Method:")
print(f"y = {equation}")


def train_polynomial_regression(X, Y, degree=2):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, Y)
    return model, poly, X_poly


model, poly, X_poly = train_polynomial_regression(X, Y)

Y_pred = model.predict(X_poly)
rmse = np.sqrt(mean_squared_error(Y, Y_pred))
print(f"\nRoot Mean Square Error (RMSE) for Polynomial Regression: {rmse:.4f}")
