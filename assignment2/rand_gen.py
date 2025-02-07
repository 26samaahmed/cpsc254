import random

arr = []
# Populate list with 10 random floating-point numbers
for _ in range(10):
  arr.append(random.uniform(-10.0, 10.0))
print(arr)