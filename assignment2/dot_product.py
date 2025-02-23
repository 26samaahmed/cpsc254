from data_loader import load_L1_csv, load_Q1_csv, load_Q2_csv
import time
import numpy as np

x_L1 = load_L1_csv()
x_Q1 = load_Q1_csv()
x_Q2 = load_Q2_csv()

def compute_dot_product(x_L1, x_Q1):
    dot_product = 0
    for i in range(len(x_L1)):
        dot_product += x_L1[i][0] * x_Q1[i][0]
    return dot_product

def compute_square(x_Q2):
    squares = []
    for i in range(len(x_Q2)):
        squares.append(x_Q2[i][0] ** 2)
    return squares

start_time = time.time()
dot_product = compute_dot_product(x_L1, x_Q1)
end_time = time.time()
print("\nDot product (element-wise multiplication) using loop: ", dot_product)
print("Time taken for dot product (loop): ", end_time - start_time)


start_time = time.time()
squares = compute_square(x_Q2)
end_time = time.time()
print("\nSquares of array elements using loop: ", squares)
print("Time taken for square computation (loop): ", end_time - start_time)


np_x_L1 = np.array(x_L1)[:, 0]
np_x_Q1 = np.array(x_Q1)[:, 0]
start_time = time.time()
np_dot_product = np.dot(np_x_L1, np_x_Q1)
end_time = time.time()
print("\nDot product using NumPy: ", np_dot_product)
print("Time taken for dot product (NumPy): ", end_time - start_time)

np_x_Q2 = np.array(x_Q2)[:, 0]
start_time = time.time()
np_square = np.square(np_x_Q2)
end_time = time.time()
print("\nSquare of array elements using NumPy: ", np_square)
print("Time taken for square computation (NumPy): ", end_time - start_time)