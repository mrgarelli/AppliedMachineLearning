import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tools.Regression import Regression

plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (20.0, 10.0)

### ACCEPT three inputs
### Two floats: the likelihood and the prior
### One list of tuples, where each tuple has two values corresponding to:
### ### ( P(Bn) , P(A|Bn) )
### ### ### Assume the list of tuples accounts for all potential values of B
### ### ### And that those values of B are all mutually exclusive.
### The list of tuples allows for the calculation of normalization constant.

### RETURN a float corresponding to the posterior probability

### YOUR ANSWER BELOW

def calc_posterior(likelihood, prior, norm_list):
	"""
	Calculate the posterior probability given likelihood,
	prior, and normalization
	
	Positional Arguments:
		likelihood -- float, between 0 and 1
		prior -- float, between 0 and 1
		norm_list -- list of tuples, each tuple has two values
			the first value corresponding to the probability of a value of "b"
			the second value corresponding to the probability of 
					a value of "a" given that value of "b"
	"""
	numerator = likelihood*prior
	denominator = sum([a*b for (a, b) in norm_list])
	return numerator/denominator

likelihood = .8
prior = .3
norm_list = [(.25 , .9), (.5, .5), (.25,.2)]
# print(calc_posterior(likelihood, prior, norm_list))
# --> 0.45714285714285713

### ACCEPT one input, a numpy array
### ### Array may be one or two dimensions

### If input is two dimensional, make sure there are more rows than columns
### ### Then prepend a column of ones for intercept term
### If input is one-dimensional, prepend a one

### RETURN a numpy array, prepared as described above,
### which is now ready for matrix multiplication with regression weights

def x_preprocess(input_x):
	"""
	Reshape the input (if needed), and prepend a "1" to every observation
	
	Positional Argument:
		input_x -- a numpy array, one- or two-dimensional
	
	Assumptions:
		Assume that if the input is two dimensional, that the observations are more numerous
			than the features, and thus, the observations should be the rows, and features the columns
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
		n = len(li)
		return np.array(li).reshape((n, 1))
	# begin matrix with ones
	def _prependOnes(mat):
		height = np.shape(mat)[0]
		toStack = [_vertMat(np.ones(height))]
		toStack.append(mat)
		return np.hstack(toStack)

	multiDimArray = len(np.shape(input_x)) > 1
	if multiDimArray:
		output = _makeVertical(input_x)
		output = _prependOnes(output)
	else: # one dimensional
		output = np.hstack((np.array([1]), input_x))
	return output

input1 = np.array([[2,3,6,9],[4,5,7,10]])
input2 = np.array([2,3,6])
input3 = np.array([[2,4],[3,5],[6,7],[9,10]])

# for i in [input1, input2, input3]:
	# print(x_preprocess(i), "\n")
	
# Outputs
	# [[ 1.  2.  4.]
	#  [ 1.  3.  5.]
	#  [ 1.  6.  7.]
	#  [ 1.  9. 10.]] 

	# [1 2 3 6] 

	# [[ 1.  2.  4.]
	#  [ 1.  3.  5.]
	#  [ 1.  6.  7.]
	#  [ 1.  9. 10.]] 

### ACCEPT four inputs:
### Two numpy arrays; an X-matrix and y-vector
### Two positive numbers, a lambda parameter, and value for sigma^2

### RETURN a 1-d numpy vector of weights.

### ASSUME your x-matrix has been preprocessed:
### observations are in rows, features in columns, and a column of 1's prepended.

### Use the above equation to calculate the MAP weights.
### ### This will involve creating the lambda matrix.
### ### The MAP weights are equal to the Ridge Regression weights

### NB: `.shape`, `np.matmul`, `np.linalg.inv`,
### `np.ones`, `np.identity` and `np.transpose` will be valuable.

### If either the "sigma_squared" or "lambda_param" are equal to 0, the return will be
### equivalent to ordinary least squares.

### YOUR ANSWER BELOW

def calculate_map_coefficients(aug_x, output_y, lambda_param, sigma_squared):
	"""
	Calculate the maximum a posteriori LR parameters
	
	Positional arguments:
		aug_x -- x-matrix of training input data, augmented with column of 1's
		output_y -- vector of training output values
		lambda_param -- positive number; lambda parameter that
			controls how heavily to penalize large coefficient values
		sigma_squared -- data noise estimate
		
	Assumptions:
		-- output_y is a vector whose length is the same as the
		number of rows in input_x
		-- input_x has more observations than it does features.
		-- lambda_param has a value greater than 0
	"""
	width_x = np.shape(aug_x)[1]
	t1 = lambda_param*sigma_squared*np.identity(width_x)
	t2 = np.matmul(aug_x.T, aug_x)
	t3 = np.linalg.inv(t1 + t2)
	t4 = aug_x.T@output_y
	coefs = np.matmul(t3, t4)
	return coefs

output_y = np.array([208500, 181500, 223500, 
							140000, 250000, 143000, 
							307000, 200000, 129900, 
							118000])
							
aug_x = np.array([[   1., 1710., 2003.],
						[   1., 1262., 1976.],
						[   1., 1786., 2001.],
						[   1., 1717., 1915.],
						[   1., 2198., 2000.],
						[   1., 1362., 1993.],
						[   1., 1694., 2004.],
						[   1., 2090., 1973.],
						[   1., 1774., 1931.],
						[   1., 1077., 1939.]])
						
lambda_param = 0.01

sigma_squared = 1000

map_coef = calculate_map_coefficients(aug_x, output_y, 
												lambda_param, sigma_squared)
												
ml_coef = calculate_map_coefficients(aug_x, output_y, 0,0)

# print(map_coef)
# --> np.array([-576.67947107   77.45913349   31.50189177])

# print(ml_coef)
#--> np.array([-2.29223802e+06  5.92536529e+01  1.20780450e+03])


### ACCEPT three inputs, all numpy arrays
### One matrix coresponding to the augmented x-matrix
### Two vectors, one of the y-target, and one of ML weights.

### RETURN the empirical data noise estimate: sigma^2. Calculated with equation given above.

### NB: "n" is the number of observations in X (rows)
### "d" is the number of features in aug_x (columns) 

### YOUR ANSWER BELOW

def estimate_data_noise(aug_x, output_y, weights):
	"""Return empirical data noise estimate \sigma^2
	Use the LR weights in the equation supplied above
	
	Positional arguments:
		aug_x -- matrix of training input data
		output_y -- vector of training output values
		weights -- vector of LR weights calculated from output_y and aug_x
		
	Assumptions:
		-- training_input_y is a vector whose length is the same as the
		number of rows in training_x
		-- input x has more observations than it does features.
		-- lambda_param has a value greater than 0
	"""
	sp = np.shape(aug_x)
	n = sp[0]
	d = sp[1]
	const = 1/(n - d)

	noise = np.zeros(n)
	for i, (slc_x, y) in enumerate(zip(aug_x, output_y)):
		mult = np.matmul(slc_x, weights)
		param = (y - mult)**2
		noise[i] = param

	noise = sum(noise)*const
	return noise

output_y = np.array([208500, 181500, 223500, 
							140000, 250000, 143000, 
							307000, 200000, 129900, 
							118000])
aug_x = np. array([[   1., 1710., 2003.],
						[   1., 1262., 1976.],
						[   1., 1786., 2001.],
						[   1., 1717., 1915.],
						[   1., 2198., 2000.],
						[   1., 1362., 1993.],
						[   1., 1694., 2004.],
						[   1., 2090., 1973.],
						[   1., 1774., 1931.],
						[   1., 1077., 1939.]])

ml_weights = calculate_map_coefficients(aug_x, output_y, 0, 0)

# print(ml_weights)
# --> [-2.29223802e+06  5.92536529e+01  1.20780450e+03]

sig2 = estimate_data_noise(aug_x, output_y, ml_weights)
# print(sig2)
#--> 1471223687.1593


### ACCEPT three inputs:
### One numpy array for the augmented x-matrix
### Two floats for sigma-squared and a lambda_param

### Calculate the covariance matrix of the posterior (capital sigma), via equation given above.
### RETURN that matrix.

### YOUR ANSWER BELOW


def calc_post_cov_mtx(aug_x, sigma_squared, lambda_param):
	"""
	Calculate the covariance of the posterior for Bayesian parameters
	
	Positional arguments:
		aug_x -- matrix of training input data; preprocessed
		sigma_squared -- estimation of sigma^2
		lambda_param -- lambda parameter that controls how heavily
		to penalize large weight values
		
	Assumptions:
		-- training_input_y is a vector whose length is the same as the
		number of rows in training_x
		-- lambda_param has a value greater than 0
	
	"""
	width_x = np.shape(aug_x)[1]
	ident = np.identity(width_x)
	t1 = lambda_param*ident
	t2 = np.matmul(aug_x.T, aug_x)/sigma_squared
	big_sigma = np.linalg.inv(t1 + t2)
	return big_sigma

output_y = np.array([208500, 181500, 223500, 
								140000, 250000, 143000, 
								307000, 200000, 129900, 
								118000])
aug_x = np. array([[   1., 1710., 2003.],
						[   1., 1262., 1976.],
						[   1., 1786., 2001.],
						[   1., 1717., 1915.],
						[   1., 2198., 2000.],
						[   1., 1362., 1993.],
						[   1., 1694., 2004.],
						[   1., 2090., 1973.],
						[   1., 1774., 1931.],
						[   1., 1077., 1939.]])
lambda_param = 0.01

ml_weights = calculate_map_coefficients(aug_x, output_y,0,0)

sigma_squared = estimate_data_noise(aug_x, output_y, ml_weights)

# print(calc_post_cov_mtx(aug_x, sigma_squared, lambda_param))
# [[ 9.99999874e+01 -1.95016334e-02 -2.48082095e-02]
#  [-1.95016334e-02  6.28700339e+01 -3.85675510e+01]
#  [-2.48082095e-02 -3.85675510e+01  5.10719826e+01]]

### ACCEPT four inputs, three numpy arrays, and one number:
### A 1-dimensional array corresponding to an augmented_x vector.
### A vector corresponding to the MAP weights, or "mu"
### A square matrix for the "big_sigma" term
### A positive number for the "sigma_squared" term

### Using the above equations

### RETURN mu_0 and sigma_squared_0 - a point estimate and variance
### for the prediction for x.

### YOUR ANSWER BELOW

def predict( aug_x, weights, big_sig, sigma_squared):
	"""
	Calculate point estimates and uncertainty for new values of x
	
	Positional Arguments:
		aug_x -- augmented matrix of observations for predictions
		weights -- MAP weights calculated from Bayesian LR
		big_sig -- The posterior covarience matrix, from Bayesian LR
		sigma_squared -- The observed uncertainty in Bayesian LR
		
	"""
	mu_0 = np.matmul(aug_x.T, weights)
	# if type(aug_x[0]) != float():
	# 	aug_x = to_pred2.astype(float)
	sigma_squared_0 = sigma_squared + np.matmul(np.matmul(aug_x.T, big_sig), aug_x)
	
	return mu_0, sigma_squared_0

output_y = np.array([208500, 181500, 223500, 
							140000, 250000, 143000, 
							307000, 200000, 129900, 
							118000])
								
aug_x = np. array([[   1., 1710., 2003.],
						[   1., 1262., 1976.],
						[   1., 1786., 2001.],
						[   1., 1717., 1915.],
						[   1., 2198., 2000.],
						[   1., 1362., 1993.],
						[   1., 1694., 2004.],
						[   1., 2090., 1973.],
						[   1., 1774., 1931.],
						[   1., 1077., 1939.]])
lambda_param = 0.01

ml_weights = calculate_map_coefficients(aug_x, output_y,0,0)

sigma_squared = estimate_data_noise(aug_x, output_y, ml_weights)

map_weights = calculate_map_coefficients(aug_x, output_y, lambda_param, sigma_squared)

big_sig = calc_post_cov_mtx(aug_x, sigma_squared, lambda_param)

to_pred2 = np.array([1,1700,1980])

# print(predict(to_pred2, map_weights, big_sig, sigma_squared))
#-->(158741.6306608729, 1593503867.9060116)

def fit_bayes_reg(input_x, output_y, lambda_param):
	
	# Ensure correct shape of X, add column of 1's for intercept
	aug_x = x_preprocess(input_x) # <----
	
	# Calculate least-squares weights
	ml_weights = calculate_map_coefficients(aug_x, output_y, 0, 0) # <----
		
	# Estimate sigma^2 from observations
	sigma_squared = estimate_data_noise(aug_x, output_y, ml_weights) # <----
	
	# Calculate MAP weights
	weights = calculate_map_coefficients(aug_x, output_y, lambda_param, sigma_squared) # <---- 

	
	# Create posterior covariance matrix
	big_sig = calc_post_cov_mtx(aug_x, sigma_squared, lambda_param) # <----
	
	return weights, big_sig

### Read in the data
tr_path = '../resource/train.csv'
data = pd.read_csv(tr_path)  

### The .head() function shows the first few lines of data for perspecitve
data.head()

input_x = data[['GrLivArea','YearBuilt']].head(100).values
output_y = data['SalePrice'].head(100).values
lambda_param = .1

mu, big_sig = fit_bayes_reg(input_x, output_y, lambda_param)

print(mu)
#--> np.array([2.10423243e-02, 4.10449281e+01, 4.22635006e+01])
print(big_sig)
# np.array([[ 9.99999861e+00, -1.75179751e-03, -2.74204060e-03],
# 			[-1.75179751e-03,  6.50420674e+00, -3.47271893e+00],
# 			[-2.74204060e-03, -3.47271893e+00,  4.60297584e+00]])