import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Dataset 1 (Linear)
X1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
Y1 = [2.5, 4.1, 5.6, 7.2, 8.8, 10.3, 11.9, 13.5, 15.0, 16.8]

# Dataset 2 (Polynomial)
X2 = np.array([-3, -2.5, -2, -1.5, -1, 0, 1, 1.5, 2, 2.5, 3]).reshape(-1, 1)
Y2 = [17.5, 12.9, 9.5, 7.2, 5.8, 5.5, 7.1, 9.7, 13.5, 18.4, 24.4]


def plot_data(X, Y, title):
    plt.scatter(X, Y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.show()

# Train Linear Regression Model
def train_linear_regression(X, Y):
    model = LinearRegression().fit(X, Y)
    return model

# Train Polynomial Regression Model
def train_polynomial_regression(X, Y, degree=2):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, Y)
    return model, poly

# Generate Linear Equation for Dataset 1
def linear_equation(X, Y):
    model = train_linear_regression(X, Y)
    intercept = model.intercept_
    coef = model.coef_[0]
    return f"y = {intercept:.4f} + {coef:.4f} * x"

# Generate Polynomial Equation for Dataset 2
def polynomial_equation(X, Y):
    model, poly = train_polynomial_regression(X, Y)
    coefficients = model.coef_
    intercept = model.intercept_
    
    feature_names = poly.get_feature_names_out(['x'])
    equation_terms = [f"{intercept:.4f}"]

    for i in range(1, len(coefficients)):
        if coefficients[i] != 0:
            term = f"{coefficients[i]:.4f} * {feature_names[i]}"
            equation_terms.append(term)

    return "y = " + " + ".join(equation_terms)


def predict_linear_value(X, Y, value):
    model = train_linear_regression(X, Y)
    return model.predict([[value]])[0]


def predict_polynomial_value(X, Y, value):
    model, poly = train_polynomial_regression(X, Y)
    transformed_value = poly.transform([[value]])
    return model.predict(transformed_value)[0]


def compute_total_error(X, Y):
    model = LinearRegression().fit(X, Y)
    Y_pred = model.predict(X)
    return np.sum(Y - Y_pred)

def compute_sum_error_squared(X, Y):
    model = LinearRegression().fit(X, Y)
    Y_pred = model.predict(X)
    return np.sum((Y - Y_pred) ** 2)

def compute_mean_square_error(X, Y):
    return compute_sum_error_squared(X, Y) / len(X)

def compute_root_mean_square_error(X, Y):
    return np.sqrt(compute_mean_square_error(X, Y))


plot_data(X1, Y1, "Dataset 1: Linear Relationship")
plot_data(X2, Y2, "Dataset 2: Polynomial Relationship")


print("\nLinear Equation for Dataset 1:")
print(linear_equation(X1, Y1))

print("\nPolynomial Equation for Dataset 2:")
print(polynomial_equation(X2, Y2))


print("\nPredicted value of Y for X = 100 (Dataset 1):", predict_linear_value(X1, Y1, 100))
print("Predicted value of Y for X = 0.5 (Dataset 2):", predict_polynomial_value(X2, Y2, 0.5))


print("\nTotal Error for Dataset 1:", compute_total_error(X1, Y1))
print("Sum of Squared Errors for Dataset 1:", compute_sum_error_squared(X1, Y1))
print("Mean Squared Error for Dataset 1:", compute_mean_square_error(X1, Y1))
print("Root Mean Squared Error for Dataset 1:", compute_root_mean_square_error(X1, Y1))

print("\nTotal Error for Dataset 2:", compute_total_error(X2, Y2))
print("Sum of Squared Errors for Dataset 2:", compute_sum_error_squared(X2, Y2))
print("Mean Squared Error for Dataset 2:", compute_mean_square_error(X2, Y2))
print("Root Mean Squared Error for Dataset 2:", compute_root_mean_square_error(X2, Y2))
