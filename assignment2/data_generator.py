import random
import json
import csv

# TODO: Create 2D array to store 100,000,000 pairs
rows1, cols1 = 2, 10
rows2, cols2 = 2, 10
matrix1 = [] # first matrix
matrix2 = [] # second matrix

def calculateY1(x):
  return 2.0 + (0.5 * x)

def calculateY2(x):
  return 2.0 + (0.5 * x) - (3 * (pow(x, 2)))

for _ in range(rows1):
    row = []
    for _ in range(cols1 // 2):
       x = random.uniform(-1000.0, +1000.0)
       pair = (calculateY1(x), x)
       row.append(pair)
    matrix1.append(row)

for _ in range(rows1):
    row = []
    for _ in range(cols1 // 2):
       x = random.uniform(-1000.0, +1000.0)
       pair = (calculateY1(x), x)
       row.append(pair)
    matrix1.append(row)

for _ in range(rows2):
    row = []
    for _ in range(cols2 // 3):
       x = random.uniform(-1000.0, +1000.0)
       values = (calculateY2(x), 0.5 * x, -3 * (pow(x, 2)))
       row.append(values)
    matrix2.append(row)

print(matrix2)

# start, end = 0, len(matrix1) - 1
# mid = (start + end) // 2
# print("First Pair", matrix1[start][start])
# print("Middle Pair",matrix1[mid][mid])
# print("Last Pair", matrix1[end][end])

for row in matrix1:
    row.sort(key=lambda pair: pair[0], reverse=True)

with open('L1.json', 'w', encoding='utf-8') as json_file:
    json.dump(matrix1, json_file, ensure_ascii=False, indent=4)

with open('L1.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['y', 'x'])
    for row in matrix1:
        writer.writerows(row) 

with open('Q2.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['y', 'x1', 'x2'])
    for row in matrix2:
        writer.writerows(row) 