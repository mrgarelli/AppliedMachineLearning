import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tools.Regression import Regression
import sys

plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (20.0, 10.0)

### Read in the data
tr_path = '../resource/train.csv'
df = pd.read_csv(tr_path)

# create a regression object from the df
reg = Regression(df)

# small dataframe for testing
dfSmall = df.head()

# Information Extraction
# print(df.head())
# print(df.columns)
# print(df.describe())
# print(list(dfSmall.index))

# __________________________________________________
# Example of multiple regression - predicting one target from many inputs
# Linear multiple-regression: y = m1*x1 + m2*x2 + b
# avoid non-relevant inputs
# avoid multicollinearity
# https://www.youtube.com/watch?v=AkBjJ6OunR4

reg.choose_params('SalePrice', ['GrLivArea', 'YearBuilt'])
reg.least_squares()

reg.generate_inputs()
reg.predict(reg.x_generated)

# determine quality of fit
reg.root_mean_squared_error()
reg.r_squared()

# check model attributes
# print(reg.y_predicted)
# print(reg.rmse)
# print(reg.rs)

### We can plot the data as follows
### Price v. living area
def plot_housing():
	Y = df['SalePrice']
	X = df['GrLivArea']
	plt.scatter(X, Y, marker = "x")
	plt.plot(reg.x_generated[:, 0], reg.y_predicted[:, 0], label='linear regression')
	plt.title("Sales Price vs. Living Area (excl. basement)")
	plt.xlabel("GrLivArea")
	plt.ylabel("SalePrice")
	plt.legend()
	# df.plot('YearBuilt', 'SalePrice', kind = 'scatter', marker = 'x')

# plot_housing()


### GRADED
### Code a function called "ridge_regression_weights"
### ACCEPT three inputs:
### Two matricies corresponding to the x inputs and y target
### and a number (int or float) for the lambda parameter

### RETURN a numpy array of regression weights

### The following must be accomplished:

### Ensure the number of rows of each the X matrix is greater than the number of columns.
### ### If not, transpose the matrix.
### Ultimately, the y input will have length n.
### Thus the x input should be in the shape n-by-p

### *Prepend* an n-by-1 column of ones to the input_x matrix

### Use the above equation to calculate the least squares weights.
### This will involve creating the lambda matrix---
### ### a p+1-by-p+1 matrix with the "lambda_param" on the diagonal
### ### p+1-by-p+1 because of the prepended "ones".

### NB: Pay close attention to the expected format of the returned
### weights. It is different / simplified from Assignment 1.

### YOUR ANSWER BELOW

def ridge_regression_weights(input_x, output_y, lambda_param):
	"""Calculate ridge regression least squares weights.
	
	Positional arguments:
		input_x -- 2-d matrix of input data
		output_y -- 1-d numpy array of target values
		lambda_param -- lambda parameter that controls how heavily
			to penalize large weight values
		
	Assumptions:
		-- output_y is a vector whose length is the same as the
		number of observations in input_x
		-- lambda_param has a value greater than 0
	"""
	# forces matrix to be verticle
	def _makeVertical(mat):
		(height, width) = mat.shape
		matHorizontal = height < width
		if matHorizontal:
			return mat.transpose()
		return mat
	# creates a vertical matrix with 1 column
	def _vertMat(li):
		'''
		returns a 1d vertical matrix from a list (or 1d numpy array)
		'''
		n = len(li)
		return np.array(li).reshape((n, 1))
	def _prependOnes(mat):
		height = np.shape(mat)[0]
		toStack = [_vertMat(np.ones(height))]
		toStack.append(mat)
		return np.hstack(toStack)

	input_x = _makeVertical(input_x)
	input_x = _prependOnes(input_x)
	
	# reogranize lambda
	lam_mat = lambda_param*np.eye(input_x.shape[1])
	lam_plus_xdotxT = lam_mat + input_x.T@input_x
	inv_lam_plus_xdotxT = np.linalg.inv(lam_plus_xdotxT)
	weights = inv_lam_plus_xdotxT@input_x.T@output_y
	return weights

training_y = np.array([208500, 181500, 223500, 
								140000, 250000, 143000, 
								307000, 200000, 129900, 
								118000])
								
training_x = np.array([[1710, 1262, 1786, 
								1717, 2198, 1362, 
								1694, 2090, 1774, 
								1077], 
								[2003, 1976, 2001, 
								1915, 2000, 1993, 
								2004, 1973, 1931, 
								1939]])
lambda_param = 10

rrw = ridge_regression_weights(training_x, training_y, lambda_param)

# print(rrw) #--> np.array([-576.67947107,   77.45913349,   31.50189177])
# print(rrw[2]) #--> 31.50189177

### Example of hiden function below:

### `hidden` takes a single number as a parameter (int or float) and returns a list of 1000 numbers
### the input must be between 0 and 50 exclusive

def hidden(hp):
	# checks if 0 >= hp >= 50
	outOfBounds = (hp <= 0, hp >= 50)
	if any(outOfBounds):
		print("[ERROR] input out of bounds")
		sys.exit()
	
	nums = np.logspace(0, 5, num = 1000)
	vals = nums**43.123985172351235134687934
	user_vals = nums** hp
	output = vals - user_vals
	
	def plot_for_understanding():
		# plt.plot(nums, label='nums')
		# plt.plot(vals, label='vals')
		# plt.plot(user_vals, label='usr_vals')
		plt.plot(output, label='out')
		plt.legend()
	# plot_for_understanding()
	
	return output

### GRADED
### Code a function called "minimize"
### ACCEPT one input: a function.

### That function will be similar to `hidden` created above and available for your exploration.
### Like 'hidden', the passed function will take a single argument, a number between 0 and 50 exclusive 
### and then, the function will return a numpy array of 1000 numbers.

### RETURN the value that makes the mean of the array returned by 'passed_func' as close to 0 as possible

### Note, you will almost certainly NOT be able to find the number that makes the mean exactly 0
### YOUR ANSWER BELOW

def minimize(passed_func):
	"""
	Find the numeric value that makes the mean of the
	output array returned from 'passed_func' as close to 0 as possible.
	
	Positional Argument:
		passed_func -- a function that takes a single number (between 0 and 50 exclusive)
			as input, and returns a list of 1000 floats.
	"""
	def mean(passed_func, inp):
		arr = passed_func(inp)
		return np.mean(arr)

	# initialize inputs
	inp = np.linspace(.01, 49.99, 5000)
	out = np.zeros(len(inp))
	# evaluate the function
	for i in range(len(inp)):
		out[i] = mean(passed_func, inp[i])

	deriv = np.gradient(out)
	def visualize():
		plt.plot(inp, label='input')
		plt.plot(out, label='output')
		plt.plot(deriv, label='derivative of output')
		plt.legend()
	# visualize()

	def nearest(array, value):
		indx = (np.abs(array - value)).argmin()
		return indx

	indx = nearest(out, 0)
	return float(inp[indx])

passed_func = hidden
# min_hidden = minimize(passed_func)
# print(round(min_hidden,4))
#--> 43.1204 (answers will vary slightly, must be close to 43.123985172351)

from sklearn.linear_model import Ridge, LinearRegression

### Note, the "alpha" parameter defines regularization strength.
### Lambda is a reserved word in `Python` -- Thus "alpha" instead

### An alpha of 0 is equivalent to least-squares regression
lr = LinearRegression()
reg = Ridge(alpha = 100000)
reg0 = Ridge(alpha = 0)

# Notice how the consistent sklearn syntax may be used to easily fit many kinds of models
for m, name in zip([lr, reg, reg0], ["LeastSquares","Ridge alpha = 100000","Ridge, alpha = 0"]):
	m.fit(df[['GrLivArea','YearBuilt']], df['SalePrice'])
	print(name, "Intercept:", m.intercept_, "Coefs:",m.coef_,"\n")

plt.show()
