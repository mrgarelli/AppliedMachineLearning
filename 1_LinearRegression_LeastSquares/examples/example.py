import numpy as np
import matplotlib.pyplot as plt
from tools.regression import \
	vertMat, \
	least_squares, \
	least_squares_def, \
	sumSq_Tot_Resid_def, \
	rmse, \
	rSquared

# input (ex. hours of sunshine)
hoursSunshine = [2, 3, 5, 7, 9]
# target/output (ex. ice cream sold)
unitSold = [4, 5, 7, 10, 15]

# put into matrix form
y = vertMat(unitSold)
x = vertMat(hoursSunshine)

# only works with a prepended x
m, b = least_squares_def(x, y)

'''
Least Squares the easy way
'''

# b_m = least_squares(x, y)

# test cases
# b_m = least_squares(np.ones(10), y)

testY = np.array(unitSold).reshape((1, 5))
b_m = least_squares(x, testY)

training_y = np.array([[208500, 181500, 223500, 140000, 250000, 
								143000, 307000, 200000, 129900, 118000]])
training_x = np.array([[1710, 1262, 1786, 1717, 2198, 
								1362, 1694, 2090, 1774, 1077], 
								[2003, 1976, 2001, 1915, 2000, 
								1993, 2004, 1973, 1931, 1939]])
w_lsq = least_squares(training_x, training_y)
# print(w_lsq)

# print(b_m)

rmse_rSq = sumSq_Tot_Resid_def(training_x, training_y, w_lsq)
print('Definition w/o linalg: ', rmse_rSq)

print(np.array([training_x[0]]).T.shape)

print(w_lsq.T.shape)

rmse, sumSq_resid = rmse(training_x, training_y, w_lsq)
rSq = rSquared(training_y, sumSq_resid)
print('With linalg: ', rmse, rSq)

'''
Plotting
'''
pts_generated = 100
x_generated = vertMat(np.linspace(0, 10, pts_generated), prepend=True)
est_definition = m*x_generated[:, 1] + b
est_numpy = b_m[1]*x_generated[:, 1] + b_m[0]

plt.scatter(x[:, 0], y[:, 0], label='data')
plt.plot(x_generated[:, 1], est_definition, label='linear regression definition')
plt.plot(x_generated[:, 1], est_numpy, label='linear regression numpy', linestyle='-.')

plt.legend()
# plt.show()