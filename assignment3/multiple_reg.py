from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import random
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

# Extract coefficients and feature names
coefficients = model.coef_
feature_names = poly.get_feature_names_out(['x1', 'x2'])

# Construct the polynomial equation
equation_terms = []
for i in range(len(coefficients)):
    if coefficients[i] != 0:
        term = f"{coefficients[i]:.4f}"
        if feature_names[i] != "1":
            term += f" * {feature_names[i]}"
        equation_terms.append(term)

polynomial_equation = " + ".join(equation_terms)


print("\nPolynomial Equation:")
print(f"y = {polynomial_equation}")

print("\nCoefficients:")
for name, coef in zip(feature_names, coefficients):
    print(f"{name}: {coef:.4f}")


value_x1 = float(random.randint(0, 1000))
value_x2 = float(random.randint(0, 1000))

def predict_polynomial_value(model, poly, value_x1, value_x2):
    transformed_input = poly.transform(pd.DataFrame([[value_x1, value_x2]], columns=['x1', 'x2']))
    return model.predict(transformed_input)[0]

predicted_y = predict_polynomial_value(model, poly, value_x1, value_x2)
print(f"\nPredicted y value for (x1={value_x1}, x2={value_x2}): {predicted_y:.2f}")

Y_pred = model.predict(X_poly)
rmse = np.sqrt(mean_squared_error(Y, Y_pred))
print(f"\nRoot Mean Square Error (RMSE): {rmse:.4f}")
