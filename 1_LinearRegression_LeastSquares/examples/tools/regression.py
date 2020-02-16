import numpy as np
import sys

def _makeHorizontal(mat):
	(height, width) = mat.shape
	matVertical = height > width
	if matVertical:
		return mat.T
	return mat

def _makeVertical(mat):
	# forces matrix to be verticle
	(height, width) = mat.shape
	matHorizontal = height < width
	if matHorizontal:
		return mat.transpose()
	return mat

def vertMat(li, prepend=False):
	'''
	returns a 1d vertical matrix from a list (or 1d numpy array)
	prepend option for the 1st column to be ones
	'''
	n = len(li)
	if (prepend == False):
		return np.array(li).reshape((n, 1))
	else:
		toStack = (vertMat(np.ones(n)), vertMat(li))
		return np.hstack(toStack)

def least_squares_def(x, y):
	'''
	Least Squares the hard way
	'''
	n = x.shape[0]

	# calculate parameters necessary to get the slope and y_intercept
	x_sq = x[:, 0]**2
	x_y = x[:, 0]*y[:, 0]

	# slope
	numerator = n*sum(x_y) - (sum(x[:, 0])*sum(y[:, 0]))
	denominator = n*sum(x_sq) - sum(x[:, 0])**2
	m = numerator/denominator
	# print(m)

	# y_intercept
	numerator = sum(y[:, 0]) - m*sum(x[:, 0])
	denominator = n
	b = numerator/denominator
	# print(b)
	return m, b

def least_squares(x, y):
	'''
	Returns the least squares coefficients for linear regression
	'''

	# run check for 2d matrices
	matricesMultiD = [len(x.shape) > 1, len(y.shape) > 1]
	if not all(matricesMultiD):
		print('[ERROR] not accepting 1 dimensional matrices')
		sys.exit()

	# run check for vertical matrices
	x = _makeVertical(x)
	y = _makeVertical(y)

	# add column of ones to x
	toStack = (vertMat(np.ones(len(x))), x)
	x = np.hstack(toStack)

	'''
	Linear Algebra solution to least squares (numpy implementation logic)
	https://www.youtube.com/watch?v=MC7l96tW8V8
	'''
	coefficients = np.linalg.solve(
		np.dot(x.T, x), 
		np.dot(x.T, y)
		)
	return coefficients

def sumSq_Tot_Resid_def(x, y, lsSq):
	'''
	inputs:
		- x input to model
		- y target of least squares model
		- lsSq coefficients as output of least squares model
	'''
	sumSq_tot = 0 # total sum of squares
	sumSq_resid = 0 # residual sum of squares
	y_mean = y.mean() # to avoid recomputing the mean
	b = lsSq[0, 0] # the y intercept
	m = lsSq[1, 0] # the slope
	n = len(x)
	for i in range(n):
		# residual is (predicted - actual)^2
		y_pred = b + m*x[i, 0]
		sumSq_resid = (y_pred - y[i, 0])**2
		# total is (actual - mean)^2
		sumSq_tot = (y[i, 0] - y_mean)**2
	# root mean squared error = squareRoot(sum of residuals / number of points)
	rmse = np.sqrt(sumSq_resid/n) # units are the units of target
	# residual errors divided by total variation from mean
	rSq = 1 - (sumSq_resid/sumSq_tot) # unitless
	return rmse, rSq

def rmse(x, y, lsSq):
	'''
	calculate the root mean squared error
	'''
	lsSq = _makeHorizontal(lsSq)
	y_pred = np.dot(x, lsSq)
	sumSq_resid = np.linalg.norm(y_pred - y)**2
	return np.sqrt(sumSq_resid/n), sumSq_resid # units are the units of target

def rSquared(y, sumSq_resid):
	sumSq_tot = np.linalg.norm(y - y.mean())**2
	return 1 - (sumSq_resid/sumSq_tot) # unitless
