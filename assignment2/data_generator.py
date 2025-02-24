import random
import json
import csv

# 1 million pairs of (x, y) for linear and quadratic functions
num_pairs = 1000000

matrix1 = []
matrix2 = []

def calculate_y1(x):
    return 2.0 + (0.5 * x)

def calculate_y2(x):
    return 2.0 + (0.5 * x) - (3 * (x ** 2))

for _ in range(num_pairs):
    x = random.uniform(-1000.0, 1000.0)
    y = calculate_y1(x)
    matrix1.append((y, x))

print('First pair in matrix1:', matrix1[0])
print('Middle pair in matrix1:', matrix1[num_pairs // 2])
print('Last pair in matrix1:', matrix1[-1])

for _ in range(num_pairs):
    x = random.uniform(-1000.0, 1000.0)
    y = calculate_y2(x)
    x1 = 0.5 * x
    x2 = -3 * (x ** 2)
    matrix2.append((y, x1, x2))


with open('L1.json', 'w', encoding='utf-8') as json_file:
    json.dump({"linear": matrix1}, json_file, ensure_ascii=False, indent=4)


with open('L1.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['y', 'x'])
    writer.writerows(matrix1)


with open('Q1.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['y', 'x'])
    for y, x1, x2 in matrix2:
        original_x = x1 * 2 
        writer.writerow([y, original_x])


with open('Q2.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['y', 'x1', 'x2'])
    writer.writerows(matrix2)
