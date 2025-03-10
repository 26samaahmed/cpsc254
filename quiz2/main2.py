def calculate_polynomial(x1, x2):
  w0 = 11
  w1 = 0
  w2 = -1
  y = w0 + (w1 * x1) + (w2 * x2)
  return y

print('Y value when x1 = 7 and x2 = 8 is: ', calculate_polynomial(7, 8))
print('Y value when x1 = 1 and x2 = 2 is: ', calculate_polynomial(1, 2))
print('Y value when x1 = 4 and x2 = 5 is: ', calculate_polynomial(4, 5))