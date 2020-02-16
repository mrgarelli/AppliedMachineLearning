import numpy as np
import pandas as pd
import sys

class Regression():
	def __init__(self, df):
		self.df = df # the pandas dataframe

		# x inputs and y target
		self.x = None
		self.y = None
		self.height = None # indices of each data column

		# generated to view predictions of model
		self.x_generated = None
		self.y_predicted = None

		# metrict to evaluate the model
		self.ssr = None # sum of squared residuals
		self.sst = None # sum squared total
		self.rmse = None # root mean squared error
		self.rs = None # r squared

		# output of the fitting
		self.coefficients = None
		self.intercept = None

	def numNulls(self):
		nulls = 0
		for colName in self.df:
			col = self.df[colName].values
			boolMx = pd.isna(col)
			nulls += sum(boolMx)
		return nulls

	def slicePanda(self, colNames):
		return self.df[colNames]

	# Inputs:
	# 		lists of strings from the dataframe to characterize xinputs and ytargets
	def choose_params(self, yParam, xParams):
		self.height = np.shape(self.df[yParam])[0] # gather data height from y target
		self.df.loc[:, yParam] = _meanCenter(self.df[yParam])
		self.y = _vertMat(self.df[yParam].values)
		toStack = [_vertMat(np.ones(len(self.df.index)))]
		for x in xParams:
			self.df.loc[:, x] = _standardize(self.df[x])
			toStack.append(_vertMat(self.df[x].values))
		self.x = np.hstack(toStack)

	def least_squares(self):
		'''
		Returns the least squares coefficients for linear regression

		Linear Algebra solution to least squares (numpy implementation logic)
		https://www.youtube.com/watch?v=MC7l96tW8V8
		'''
		w_ls = np.linalg.solve(
			np.dot(self.x.T, self.x), 
			np.dot(self.x.T, self.y)
			)
		self.coefficients = w_ls[1:]
		self.intercept = w_ls[:1]

	def predict(self, generated_inputs):
		self.y_predicted = self.intercept + np.dot(generated_inputs, self.coefficients)

	def generate_inputs(self):
		toStack = []
		for x in self.x[:, 1:].T: # iterates over the columns in input matrix, ignores ones
			singleCol = _vertMat(np.linspace(min(x), max(x), self.height))
			toStack.append(singleCol)
		self.x_generated = np.hstack(toStack)

	def root_mean_squared_error(self):
		'''
		calculate the root mean squared error
		'''
		if self.y_predicted is None:
			print('[ERROR] must run predict to get y_predicted attribute')
			sys.exit()
		self.ssr = np.linalg.norm(self.y_predicted - self.y)**2
		self.rmse = np.sqrt(self.ssr/self.height) # units are the units of target

	def r_squared(self):
		self.sst = np.linalg.norm(self.y - self.y.mean())**2
		self.rs = 1 - (self.ssr/self.sst) # unitless

def _vertMat(li):
	'''
	returns a 1d vertical matrix from a list (or 1d numpy array)
	'''
	n = len(li)
	return np.array(li).reshape((n, 1))

def _meanCenter(lst):
	return lst - np.mean(lst)

def _standardize(lst):
	return _meanCenter(lst)/np.std(lst)

def _makeVertical(mat):
	# forces matrix to be verticle
	(height, width) = mat.shape
	matHorizontal = height < width
	if matHorizontal:
		return mat.transpose()
	return mat