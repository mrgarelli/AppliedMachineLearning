import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def find_splits(col):
	"""
	Calculate and return all possible split values given a column of numeric data

	Positional argument:
	  col -- a 1-dimensional numpy array, corresponding to a numeric
			predictor variable.

	"""
	  
	un = np.unique(col)
	splits = (un[:-1] + un[1:])/2
	return splits

col = np.array([0.5, 1. , 3. , 2. , 3. , 3.5, 3.6, 4. , 4.5, 4.7])
splits  = find_splits(col)
# print(splits) # --> np.array([0.75, 1.5, 2.5, 3.25, 3.55, 3.8, 4.25, 4.6])

def entropy(class1_n, class2_n):
	# If all of one category, log2(0) does not exist,
	# and entropy = 0
	if (class1_n == 0) or (class2_n == 0):
		return 0

	# Find total number of observations 
	total = class1_n + class2_n

	# find proportion of both classes
	class1_proprtion = class1_n/total
	class2_proportion = class2_n/total

	# implement entropy function
	return  sum([-1 * prop * np.log2(prop)
					for prop in [class1_proprtion, class2_proportion] ])

# print(entropy(3,1))

### GRADED
### Code a function called `ent_from_split`
### ACCEPT three inputs:
### 1. A numpy array of values
### 2. A value on which to split the values in the first array, (into two groups; <= and >)
### 3. Labels for the observations corresponding to each value in the first array.
### ### Assume the labels are "0"s and "1"s

### RETURN the entropy resulting from that split: a float between 0 and 1.

### Feel free to use the `entropy()` function defined above
### YOUR ANSWER BELOW

def ent_from_split(col, split_value, labels):
	total = len(col)
	# divide class arrays
	d1 = labels[col <= split_value]
	d2 = labels[col > split_value]
	# get class counts for each division
	d1c1 = sum(d1)
	d1c0 = len(d1) - d1c1
	d2c1 = sum(d2)
	d2c0 = len(d2) - d2c1
	# entropy of each class in same division
	d1_entropy = entropy(d1c0, d1c1)
	d2_entropy = entropy(d2c0, d2c1)
	# get proportion of each division
	d1_proportion = len(d1)/total
	d2_proportion = len(d2)/total
	ent = d1_proportion*d1_entropy + d2_proportion*d2_entropy
	return ent

col = np.array([1,1,2,2,3,3,4])
split = 2.5
labels = np.array([0,1,0,0,1,0,1])

# print(ent_from_split(col, split, labels)) # --> 0.8571428571428571

### Code a function called `pred_from_split`
### ACCEPT four inputs:
### 1. a numpy array of observations
### 2. a numpy array of labels: 0's and 1's
### 3. a column index
### 4. a value to split that column specified by the index

### RETURN a tuple of (left_pred, right_pred) where:
### left_pred is the majority class of labels where observations are <= split_value
### right_pred is the majority class of labels where observations are > split_value

### If the split yeilds equal number of observations of each class in BOTH nodes,
### ### let both `left_pred` and `right_pred` be 1.
### If the split yeilds equal number of observations of each class in ONLY ONE node,
### ### predict the opposite of the other node. e.g.

### ### node 1    |   node 2
### ###  c1  | c2 |  c1 | c2
### ###  5  | 4   |  3  |  3

### The prediction for node 1 would be "class 1".
### Because of the equal numbers of each class in node 2,
### the prediction for node 2 would be the opposite of the node 1 prediction.
### e.g. the prediction for node 2 would be "class 2"

def pred_from_split(X, y, col_idx, split_value):
	"""
	Return predictions for the nodes defined by the given split.
	Positional argument:
		X -- a 2-dimensional numpy array of predictor variable observations.
			rows are observations, columns are features.
		y -- a 1-dimensional numpy array of labels, associated with observations
			in X.
		col_idx -- an integer index, such that X[:,col_idx] yeilds all the observations
			of a single feature.
		split_value -- a numeric split, such that the values of X[:,col_idx] that are
			<= split_value are in the left node. Those > split_value are in the right node.
	"""
	def mostCommon(ctr): return ctr.most_common()[0][0]
	def hasEqualClasses(ctr):
		if len(ctr.most_common()) == 1: return False
		return ctr.most_common()[0][1] == ctr.most_common()[1][1]
	def flip(bol):
		if bol == 0: return 1
		return 0
	# divide class arrays
	arr = X[:, col_idx]
	indices = np.argsort(arr)
	col = arr[indices]
	labels = y[indices]
	lC = Counter(labels[col <= split_value])
	rC = Counter(labels[col > split_value])
	if hasEqualClasses(lC) & hasEqualClasses(rC): return (1, 1)
	if hasEqualClasses(lC): return (flip(mostCommon(rC)), mostCommon(rC))
	if hasEqualClasses(rC): return (mostCommon(lC), flip(mostCommon(lC)))
	return (mostCommon(lC), mostCommon(rC))

X = np.array([[0.5, 3. ], [1.,  2. ], [3.,  0.5],
				  [2.,  3. ], [3.,  4. ]])
y = np.array([ 1, 1, 0, 0, 1])
col_idx = 0
split_value = 1.5
pred_at_nodes = pred_from_split(X, y, col_idx, split_value)
print(pred_at_nodes) # --> (1, 0)


### GRADED
### Code a function called "simple_binary_tree_predict"
### ACCEPT five inputs:
### 1. A numpy array of observations
### 2. A column index
### 3. A value to split the column specified by the index
### 4/5. Two values, 1 or 0, denoting the predictions at left and right nodes

### RETURN a numpy array of predictions for each observation

### Predictions are created for each row in x:
### 1. For a row in X, find the value in the "col_idx" column
### 2. Compare to "split_value"
### 3. If <= "split_value", predict "left_pred"
### 4. Else predict "right_pred"

### YOUR ANSWER BELOW

def simple_binary_tree_predict(X, col_idx, split_value, left_pred, right_pred):
	"""
	Create an array of predictions built from: observations in one column of X,
			a given split value, and given predictions for when observations
			are less-than-or-equal-to that split or greater-than that split value
	Positional arguments:
			X -- a 2-dimensional numpy array of predictor variable observations.
					rows are observations, columns are different features
			col_idx -- an integer index, such that X[:,col_idx] yeilds all the observations
					in a single feature.
			split_value -- a numeric split, such that the values of X[:,col_idx] that are
					<= split_value are in the left node, and those > are in the right node.   
			left_pred -- class (0 or 1), that is predicted when observations
					are less-than-or-equal-to the split value
			right_pred -- class (0 or 1), that is predicted when observations
					are greater-than the split value
	"""
	obs = X[:, col_idx]
	predictions = np.array([left_pred if o <= split_value else right_pred for o in obs])
	return predictions

X = np.array([[0.5, 3. ], [1.,  2. ], [3.,  0.5],
							[2.,  3. ], [3.,  4. ]])
col_idx = 0
split_value = 1.5
left_pred = 1
right_pred = 0

preds = simple_binary_tree_predict(X, col_idx, split_value, left_pred, right_pred)
# print(preds) #--> np.array([1,1,0,0,0])
