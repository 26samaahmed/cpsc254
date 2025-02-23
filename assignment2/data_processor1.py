from data_loader import load_Q1_csv, load_Q2_csv
import random
import numpy as np
import pandas as pd


def calculate_mean(data):
    mean = [0] * len(data[0])  # Initialize mean list depending on the number of columns
    for row in data:
        for i in range(len(row)):
            mean[i] += row[i]
    for i in range(len(mean)):
        mean[i] /= len(data)
    return mean


def calculate_std_dev(data, mean):
    std_dev = [0] * len(data[0]) 
    for row in data:
        for i in range(len(row)):
            std_dev[i] += (row[i] - mean[i]) ** 2
    for i in range(len(std_dev)):
        std_dev[i] = (std_dev[i] / len(data)) ** 0.5
    return std_dev

def identify_outliers(data, mean, std_dev):
    outliers = []
    for row in data:
        for i in range(len(row)):
            if row[i] < mean[i] - 2 * std_dev[i] or row[i] > mean[i] + 2 * std_dev[i]:
                outliers.append(row)
                break 
    return outliers

# Remove Outliers using normal loops
def remove_outliers(data, outliers):
    new_data = []
    for row in data:
        is_outlier = False
        for outlier in outliers:
            if row == outlier:
                is_outlier = True
                break
        if not is_outlier:
            new_data.append(row)
    return new_data

# Normalize data to the range [0, 1] and adjust mean to 0
def normalize_data(data):
    min_vals = []
    max_vals = []
    for i in range(len(data[0])):
        min_val = float('inf')
        max_val = float('-inf')
        for row in data:
            if row[i] < min_val:
                min_val = row[i]
            if row[i] > max_val:
                max_val = row[i]
        min_vals.append(min_val)
        max_vals.append(max_val)
    
    # Normalize each value in the data
    normalized = []
    for row in data:
        normalized_row = []
        for i in range(len(row)):
            normalized_value = (row[i] - min_vals[i]) / (max_vals[i] - min_vals[i])
            normalized_row.append(normalized_value)
        normalized.append(normalized_row)

    # Adjust the mean to 0 by subtracting the mean of each column
    column_means = []
    for i in range(len(normalized[0])):
        column_sum = 0
        for row in normalized:
            column_sum += row[i]
        column_means.append(column_sum / len(normalized))

    for row in normalized:
        for i in range(len(row)):
            row[i] -= column_means[i]
    
    return normalized

def display_random_values(data, n=10):
    """Displays n random values from the data."""
    for _ in range(n):
        random_index = random.randint(0, len(data) - 1)
        print(data[random_index])

q1_data = load_Q1_csv()
q2_data = load_Q2_csv()

q1_mean = calculate_mean(q1_data)
q1_std_dev = calculate_std_dev(q1_data, q1_mean)
q2_mean = calculate_mean(q2_data)
q2_std_dev = calculate_std_dev(q2_data, q2_mean)

outliers_q1 = identify_outliers(q1_data, q1_mean, q1_std_dev)
q1_data = remove_outliers(q1_data, outliers_q1)
outliers_q2 = identify_outliers(q2_data, q2_mean, q2_std_dev)
q2_data = remove_outliers(q2_data, outliers_q2)

print("Outliers for Q1:", outliers_q1)
print("Outliers for Q2:", outliers_q2)

print("\nRandom values from Q1 after removing outliers:")
display_random_values(q1_data)
print("\nRandom values from Q2 after removing outliers:")
display_random_values(q2_data)

q1_data_normalized = normalize_data(q1_data)
q2_data_normalized = normalize_data(q2_data)

q1_df = pd.DataFrame(q1_data_normalized, columns=['y', 'x'])
q2_df = pd.DataFrame(q2_data_normalized, columns=['y', 'x1', 'x2'])

print("\nRandom values from Q1 DataFrame:")
display_random_values(q1_df.values)
print("\nRandom values from Q2 DataFrame:")
display_random_values(q2_df.values)

q1_np_arr = np.array(q1_data_normalized)
q2_np_arr = np.array(q2_data_normalized)

print("\nRandom values from Q1 NumPy array:")
display_random_values(q1_np_arr)
print("\nRandom values from Q2 NumPy array:")
display_random_values(q2_np_arr)

X1_Data = []
Y1_Data = []
for row in q1_np_arr:
    X1_Data.append(row[1])
    Y1_Data.append(row[0])

X2_Data = []
Y2_Data = []
for row in q2_np_arr:
    X2_Data.append(row[1:])
    Y2_Data.append(row[0])
