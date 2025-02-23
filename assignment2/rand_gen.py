import random

# Populate list with 10 random floating-point numbers
arr = []
def random_generator(arr):
  for _ in range(10):
    arr.append(random.uniform(-10.0, 10.0))
  return arr

print(random_generator(arr))