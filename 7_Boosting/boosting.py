import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

### GRADED
### Code a function called 'default_weights
### ACCEPT a single integer, `n`,  as input
### RETURN default weights; a numpy array of lenth n, where each value is equal to 1/n

def default_weights(n):
	"""
	Create the default list of weights, a numpy array of length n
	with each value equal to 1/n
	"""
	return np.full((n), 1/n)

n = 10
dw = default_weights(n)
# print(dw) #--> np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
			

def boot_strap_selection(X, y, weights):
	"""
	Create and return a boot-strapped sample of the given data,
	According to the provided weights.
	Positional Arguments:
			X -- a numpy array, corresponding to the matrix of x-observations
			y -- a numpy array, corresponding to a vector of y-labels
					All either 0 or 1
			weights -- a numpy array, corresponding to the rate at which the observations
					should be sampled for the boot-strap. 

	Example: 
			X = np.array([[1,1],[2,2],[3,3],[4,4],[5,5]])
			y = np.array([1,0,1,0,1])
			weights = np.array([.35,.1,.1,.35,.1])
			
			print(boot_strap_selection(X,y, weights))
	"""
	
	# Take random sample of indicies, with replacement
	bss_indicies = np.random.choice(range(len(y)), size = len(y), p = weights)
	
	# Subset arrays with indicies
	return X[bss_indicies,:], y[bss_indicies]

### Example of use
X = np.array([[1,1],[2,2],[3,3],[4,4],[5,5]])
y = np.array([1,0,1,0,1])
weights = np.array([.35,.1,.1,.35,.1])

# print(boot_strap_selection(X,y, weights))    
'''
### Actual results will vary
np.array([[4, 4],
				[2, 2],
				[4, 4],
				[5, 5],
				[5, 5]]),
np.array([0, 0, 0, 1, 1])
'''


### GRADED
### Code a function called `calc_epsilon` 
### ACCEPT three inputs:
### 1. The True labels
### 2. The Predicted labels
### 3. The current Weights

### RETURN the epsilon value, calculated according to the above equation.
### ### Will be a float between 0 and 1

### The epsilon is the sum of the weights where the true-label DOES NOT EQUAL the predicted-label

### YOUR ANSWER BELOW

def calc_epsilon(y_true, y_pred, weights):
	"""
	Calculate the value of epsilon, given the above equation 
	Positional Arguments:
			y_true -- An np.array of 1's and 0's corresponding to whether each observation is
					a member of class 1 or class 2
			y_pred -- An np.array of 1's and 0's corresponding to whether each observation was
					predicted to be a member of class 1 or class 2
			weights -- An np.array of floats corresponding to each observation's weight. 
					All the weights will sum up to 1.
	Assumptions:
			Assume both the true labels and the predictions are both all 0's and 1's.
	"""
	return sum([w if t!=p else 0 for t, p, w in zip(y_true, y_pred, weights)])

y_true = np.array([1,0,1,1,0])
y_pred = np.array([0,0,0,1,0])
weights = np.array([.4,.4,.1,.05,.05])

ep = calc_epsilon(y_true, y_pred, weights)

# print(ep) # --> .5

### GRADED
### Code a function called `calc_alpha`
### ACCEPT a non-negative float (epsilon) as input
### RETURN the alpha (float) calculated using the equation above.
### HOWEVER, if epsilon equals 0, return np.inf

### NB: np.log() calculates the natural log

### YOUR ANSWER BELOW

def calc_alpha(epsilon):
	"""
	Calculate the alpha value given the epsilon observed from a model
	Positional Argument:
			epsilon -- The epsilon value calculated from a particular model
	"""
	if epsilon==0: return np.inf
	return .5*np.log((1 - epsilon)/epsilon)

ep = .4
# alpha = calc_alpha(ep)
# print(alpha) # --> 0.2027325540540821

### Code a function "update_weights"
### ACCEPT four inputs:
### 1. A numpy array of a weight vector
### 2. An alpha value (float)
### 3/4. numpy arrays of true labels and predicted labels vectors.

### NB: Labels will need to be converted from 0s and 1s to -1s and 1s

### RETURN an updated array of weights, according to equation above.

### YOUR ANSWER BELOW
def update_weights(weights, alpha, y_true, y_pred):
	"""
	Create an updated vector of weights according to the above equations
	Positional Arguments:
			weights -- a 1-d numpy array of positive floats, corresponding to 
					observation weights
			alpha -- a positive float
			y_true -- a 1-d numpy array of true labels, all 0s and 1s
			y_pred -- a 1-d numpy array of labels predicted by the last model;
						all 0s and 1s. 
	"""
	# changes 0s to -1s and returns an array of floats instead
	def formatLabels(arr):
		return np.array([-1 if l == 0 else 1 for l in arr])

	y_true = formatLabels(y_true)
	y_pred = formatLabels(y_pred)
	weights = weights

	w_hat = weights*np.exp(-1*alpha*y_true*y_pred)
	updatedWeights = np.true_divide(w_hat, sum(w_hat))

	print(w_hat)
	print(sum(w_hat))
	print()
	print(updatedWeights)
	return updatedWeights

y_true = np.array([1,0,1,1,0])
y_pred = np.array([0,0,1,1,1])
weights = np.array([.4,.4,.1,.05,.05])
alpha = 0.10033534773107562

# print(update_weights(weights, alpha, y_true, y_pred))
#-->np.array([0.44444444 0.36363636 0.09090909 0.04545455 0.05555556])


### ACCEPT two inputs:
### 1. a 2-d numpy array of x-obervations
### 2. a dictionary that contains classifiers and alphas (described more below and above)

### Combine the models as in the manner described in the equation above
### to create predictions for the observations.

### RETURN a 1-d numpy array of observations (all 0s and 1s)

### YOUR ANSWER BELOW

def predict(X, est_dict):
	"""
	Create a np.array list of predictions for all of the observations in x,
	according to the above equation.
	Positional Arguments:
			X -- a 2-d numpy array of X observations. Features in columns, 
					observations in rows.
			est_dict -- a dictionary consists of keys 0 through n with tuples as values
					The tuples will be (<mod>, alpha), where alpha is a float, and 
					<mod> is a sklearn DecisionTreeClassifier
	Assumptions:
			The models in the `est-dict` tuple will return 0s and 1s.
					HOWEVER, the prediction equation depends upon predictions
					of -1s and 1s.
					FINALLY, the returned predictions should be 0s and 1s.            
	"""
	total_prediction = np.zeros(X.shape[0])
	for k in est_dict:
		# make predictions using the model
		labels = est_dict[k][0].predict(X)
		# format labels
		labels[labels==0] = -1
		alpha = est_dict[k][1]
		weighted_labels = labels*alpha
		total_prediction += weighted_labels
	total_prediction[total_prediction > 0] = 1
	total_prediction[total_prediction < 0] = 0
	return total_prediction


### Our example dataset, inspired from lecture
pts = [[.5, 3,1],[1,2,1],[3,.5,0],[2,3,0],[3,4,1],
	[3.5,2.5,0],[3.6,4.7,1],[4,4.2,1],[4.5,2,0],[4.7,4.5,0]]

df = pd.DataFrame(pts, columns = ['x','y','classification'])

### split out X and labels
X = df[['x','y']]
y = df['classification']
### Split data in half
X1 = X.iloc[:len(X.index)//2, :]
X2 = X.iloc[len(X.index)//2:, :]

y1 = y[:len(y)//2]
y2 = y[len(X)//2:]


### Fit classifiers to both sets of data, save to dictionary:

### Tree-creator helper function
def simple_tree():
	return DecisionTreeClassifier(criterion = 'entropy', max_depth= 1)
            
tree_dict = {}

tree1 = simple_tree()
tree1.fit(X1,y1)
print("threshold:", tree1.tree_.threshold[0], "feature:", tree1.tree_.feature[0])

### made up alpha, for example
alpha1 = .6
tree_dict[1] = (tree1, alpha1)

tree2 = simple_tree()
tree2.fit(X2,y2)
print("threshold:", tree2.tree_.threshold[0], "feature:" ,tree2.tree_.feature[0])

### made up alpha, again
alpha2 = .35
tree_dict[2] = (tree2, alpha2)

print(predict(X, tree_dict))
#--> np.array([1., 1., 0., 0., 0., 0., 0., 0., 0., 0.])

###############################
### For Further Checking of your function:
### The sum of predictions from the two models should be:

# If tree2 splits on feature 1:
# np.array([ 0.25  0.25 -0.95 -0.95 -0.25 -0.95 -0.25 -0.25 -0.95 -0.25])

# If tree2 splits on feature 0:
# np.array([ 0.95  0.95 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.95 -0.95])
###############################


def simple_adaboost_fit(X,y, n_estimators):
	"""
	Positional arguments :
			X -- a numpy array of numeric observations:
					rows are observations, columns are features
			y -- a numpy array of binary labels:
					*Assume labels are 1 for "True" and 0 for "False"*
			estimator -- a model capable of binary classification, implementing
					the `.fit()` and `.predict()` methods.
			n_estimators -- The number of estimators to fit.
	Steps:
			1. Create probability weights for selection during boot-straping.
			2. Create boot-strap sample of observations according to weights
			3. Fit estimator model with boot-strap sample.
			4. Calculate model error: epsilon
			5. Calculate alpha to associate with model
			6. Re-calculate probability weights
			7. Repeat 2-6 unil creation of n_estimators models. 
	"""
	
	def simple_tree():
		return DecisionTreeClassifier(criterion = 'entropy', max_depth= 1)
	
	# Create default weights array where all are equal to 1/n
	weights = default_weights(len(y)) ### <------
	
	est_dict = {}
	for i in range(n_estimators):
		# Create bootstrap sample
		bs_X, bs_y = boot_strap_selection(X, y, weights)
		
		mod = simple_tree()
		mod.fit(bs_X, bs_y)
		
		# Note: Predicting on all values of X, NOT boot-strap
		preds = mod.predict(X)
		
		epsilon = calc_epsilon(y, preds, weights) ### <------
		alpha = calc_alpha(epsilon) ### <------
		
		# Note that the i+1-th model will be keyed to the int i,
		# and will store a tuple of the fit model and the alpha value
		est_dict[i] = (mod, alpha)
		
		weights = update_weights(weights, alpha, y, preds) ### <------
	
	return est_dict 