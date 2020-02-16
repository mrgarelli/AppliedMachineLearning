import numpy as np

arrs = [np.array([0, 1, 2, 3, 4, 5]), np.array([6, 7, 8, 9, 10, 11])]
mtrx = np.vstack(arrs)
# print(mtrx)

arr = np.array([0, 1, 2, 3, 4, 5])

# print()
# print('transposed: ')
# mtrx = mtrx.T
# print(mtrx)

# print()
# print('transposed again: ')
# mtrx = mtrx.T
# print(mtrx)

# print()
# print('reshaped: ')
# mtrx = mtrx.reshape(-1, 1)
# print(mtrx)

print(arr)

print(arr.T)

print(arr.reshape(-1, 1))