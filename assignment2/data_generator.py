import random

# TODO: Create 2D array to store 100,000,000 pairs
rows, cols = (3,3)
matrix = [[0] * cols] * rows

def calculateY(x):
  return 2.0 + (0.5 * x)

for j in range(rows):
  for i in range(cols):
    x = random.uniform(-1000.0, +1000.0)
    matrix[i][j] = calculateY(x)


json_file = open("L1.json")
#TODO: Save data to json file in linear order


json_file = open("L1.csv")
#TODO: Save comma seperated data to csv file