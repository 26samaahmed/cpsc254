import random
import json
import csv

num_pairs = 15

matrix1 = []
matrix2 = []

def calculate_y1(x):
    return 2.0 + (0.5 * x)

def calculate_y2(x):
    return 2.0 + (0.5 * x) - (3 * (x ** 2))

for _ in range(num_pairs):
    x = random.uniform(-10.0, 10.0)
    y = calculate_y1(x)
    matrix1.append((y, x))


for _ in range(num_pairs):
    x = random.uniform(-10.0, 10.0)
    y = calculate_y2(x)
    x1 = 0.5 * x
    x2 = -3 * (x ** 2)
    matrix2.append((y, x1, x2))

matrix1.sort(key=lambda pair: pair[0], reverse=True)

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
