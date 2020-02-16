import numpy as np

"""Average first and last element of a 1-D array"""
def my_func(a): return (a[0] + a[-1]) * 0.5

b = np.array([[1,2,3], [4,5,6], [7,8,9]])
print(b)
print()
print('0 axis: ', np.apply_along_axis(my_func, 0, b))
print('1 axis: ', np.apply_along_axis(my_func, 1, b))
