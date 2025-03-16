from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import os


csv_path = '../assignment2/Q2.csv'
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Error: The file '{csv_path}' was not found. Please check the path.")

load_Q2_csv = pd.read_csv(csv_path)
X = load_Q2_csv[['x1', 'x2']]
Y = load_Q2_csv['y']

# Fit Polynomial Regression Model
def train_polynomial_regression(X, Y, degree=2):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, Y)
    return model, poly, X_poly

model, poly, X_poly = train_polynomial_regression(X, Y)

# Display Coefficients
print("\nCoefficients:")
print(model.coef_)

# Compute the RMSE
Y_pred = model.predict(X_poly)
rmse = np.sqrt(mean_squared_error(Y, Y_pred))
print(f"\nRoot Mean Square Error (RMSE): {rmse:.4f}")

# User Input for Prediction
value_x1 = float(input("\nEnter x1 value: "))
value_x2 = float(input("Enter x2 value: "))

def predict_polynomial_value(model, poly, value_x1, value_x2):
    transformed_input = poly.transform(pd.DataFrame([[value_x1, value_x2]], columns=['x1', 'x2']))
    return model.predict(transformed_input)[0]

predicted_y = predict_polynomial_value(model, poly, value_x1, value_x2)
print(f"\nPredicted y value for (x1={value_x1}, x2={value_x2}): {predicted_y:.2f}")
