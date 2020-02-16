import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.figsize'] = (20.0, 10.0)

from NullsCalc.NullsCalc import NullsCalc as NullsCalc

# Load the data into a `pandas` DataFrame object
tr_path = '../rsrc/train.csv'
titanic_df = pd.read_csv(tr_path)

# Examine head of df
# print(titanic_df.head(7))
# print(titanic_df.columns)

### 1. Drop all of the columns in `titanic_df` which are filled more than 50% with nulls.
### 2. If a column has fewer than 10 missing values:
### ### Drop all of the records with missing data in that column.

### After performing the above drops, what is the shape of the DataFrame?
### Assign ints to `row` and `cols` below corresponding to the *remaining* number of rows / columns

### YOUR ANSWER BELOW
nCalc = NullsCalc(titanic_df)

# print(nCalc.percentNulls)
# print(nCalc.numNulls['Age'])

nCalc.dropColNulls(.5)
nCalc.dropRowNulls(10)

# print(titanic_df.shape)

### Drop irrelevant categories
titanic_df.drop(
	# ['Ticket','Cabin', 'PassengerId', 'Name'],
	['Ticket', 'PassengerId', 'Name'],
	axis=1, inplace=True
	)

# subset of dataframe where Embarked is not null
titanic_df = titanic_df.loc[titanic_df['Embarked'].notnull(), :]

### Drop "Survived" for purposes of KNN imputation:
y_target = titanic_df.Survived
titanic_knn = titanic_df.drop(['Survived'], axis = 1)  
# print(titanic_knn.head())

### Adding dummy variables for categorical vars
to_dummy = ['Sex','Embarked']
titanic_knn = pd.get_dummies(
	titanic_knn,
	prefix = to_dummy,
	columns = to_dummy,
	drop_first = True
	)

# print head after dummy variables convert sex, embarked to binary arrays
# print(titanic_knn.head())

### Splitting data - on whether or not "Age" is specified.

# Training data -- "Age" Not null; "Age" as target
train = titanic_knn[titanic_knn.Age.notnull()]
X_train = train.drop(['Age'], axis = 1)
y_train = train.Age


############ Data to impute
# Where Age is null; Remove completely-null "Age" column.
# basically a dataframe of everything else where age is null
impute = titanic_knn[
	titanic_knn.Age.isnull()
	].drop(['Age'], axis = 1)
# print("Data to Impute")
# print(impute.head(3))

# import algorithm
from sklearn.neighbors import KNeighborsRegressor

# Instantiate
knr = KNeighborsRegressor()

# Fit
knr.fit(X_train, y_train)

# Create Predictions
imputed_ages = knr.predict(impute)

# Add to Df
impute['Age'] = imputed_ages
# print("\nImputed Ages")
# print(impute.head(3))

# Re-combine dataframes
titanic_imputed = pd.concat([train, impute], sort = False, axis = 0)

# Return to original order - to match back up with "Survived"
titanic_imputed.sort_index(inplace = True)
# print("Shape with imputed values:", titanic_imputed.shape)
# print("Shape before imputation:", titanic_knn.shape)
titanic_imputed.head(7)

import itertools
# Lists of categorical v. numeric features
categorical = ['Pclass','Sex','Embarked']
numeric = ['Age','SibSp','Parch','Fare']

# Create all pairs of categorical variables, look at distributions
cat_combos = list(itertools.combinations(categorical, 2))
# print("All Combos or categorical vars: \n",cat_combos, "\n")
for row, col in cat_combos:
	# print("Row Percents: \n",pd.crosstab(titanic_df[row], titanic_df[col], normalize="index"), "\n")
	# print("Column Percents: \n", pd.crosstab(titanic_df[row], titanic_df[col], normalize="columns"),"\n---------------\n")
	pass

import seaborn as sns
sns.heatmap(titanic_df[numeric].corr(), cmap = "coolwarm")
# plt.show()

### GRADED
### Follow directions given above
### YOUR ANSWER BELOW

def prepare_data(input_x, target_y):
	"""
	Confirm dimensions of x and y, transpose if appropriate;
	Add column of ones to x;
	Ensure y consists of 1's and -1's;
	Create weights array of all 0s
	Arguments:
		input_x - a numpy array 
		target_y - a numpy array
	Returns:
		prepared_x -- a 2-d numpy array; first column consists of 1's,
			more rows than columns
		prepared_y -- a numpy array consisting only of 1s and -1s
		initial_w -- a 1-d numpy array consisting of "d+1" 0s, where
			"d+1" is the number of columns in "prepared_x"
	Assumptions:
		Assume that there are more observations than features in `input_x`
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

	input_x = _makeVertical(input_x)

	prepared_y = target_y
	prepared_y[prepared_y == 0] = -1
	prepared_x = _prependOnes(input_x)

	width = prepared_x.shape[1]
	initial_w = np.zeros(width)
	
	return prepared_x, prepared_y, initial_w

x = np.array([[1,2,3,4],[11,12,13,14]])
y = np.array([1,0,1,1])
x,y,w = prepare_data(x,y)

# print(x)   #--> array([[ 1,  1, 11],
							# [ 1,  2, 12],
							# [ 1,  3, 13],
							# [ 1,  4, 14]])
							
# print(y) #--> array([1, -1, 1, 1])

# print(w) #--> array([0., 0., 0.])

def sigmoid_single(x, y, w):
	"""
	Obtain the value of a Sigmoid using training data.
	
	Arguments:
		x - a vector of length d
		y - either 1, or -1
		w - a vector of length d
	
	"""
	chk = y*(x.T@w)
	if chk > 709.782:
		return 1
	val = np.exp(chk)
	return val/(1 + val)

x = np.array([23.0,75])
y = -1
w = np.array([2,-.5])
sig = sigmoid_single(x, y, w)

# print(sig) #--> 0.0002034269780552065

x2 = np.array([ 1. , 22., 0. , 1. , 7.25 , 0. , 3. , 1. , 1.])
w2 = np.array([ -10.45 , -376.7215 , -0.85, -10.5 , 212.425475 , -1.1, -36.25 , -17.95 , -7.1])
y2 = -1
sig2 = sigmoid_single(x2,y2,w2)

# print(sig2) #--> 1


def to_sum(x,y,w):
	"""
	Obtain the value of the function that will eventually be summed to 
	find the gradient of the log-likelihood.
	
	Arguments:
		x - a vector of length d
		y - either 1, or -1
		w - a vector of length d
		
	"""
	return (1 - sigmoid_single(x, y, w))*y*x

x = np.array([23.0,75])
y = -1
w = np.array([.1,-.2])
# print(to_sum(x,y,w)) # --> array([-7.01756737e-05, -2.28833719e-04])

def sum_all(x_input, y_target, w):
	"""
	Obtain and return the gradient of the log-likelihood
	Arguments:
		x_input - *preprocessed* an array of shape n-by-d
		y_target - *preprocessed* a vector of length n
		w - a vector of length d
	"""
	tot = np.zeros(len(w))
	for x, y in zip(x_input, y_target):
		val = to_sum(x, y, w)
		tot += val
	return tot

x = np.array([[1,22,7.25],[1,38,71.2833]])
y = np.array([-1,1])
w = np.array([.1,-.2, .5])
# print(sum_all(x,y,w)) #--> array([-0.33737816, -7.42231958, -2.44599168])

def update_w(x_input, y_target, w, eta):
	"""Obtain and return updated Logistic Regression weights
	Arguments:
		x_input - *preprocessed* an array of shape n-by-d
		y_target - *preprocessed* a vector of length n
		w - a vector of length d
		eta - a float, positive, close to 0
	"""
	return w + eta*sum_all(x_input, y_target, w)

x = np.array([[1,22,7.25],[1,38,71.2833]])
y = np.array([-1,1])
w = np.array([.1,-.2, .5])
eta = .1

# print(update_w(x,y,w, eta)) #--> array([ 0.06626218, -0.94223196,  0.25540083])

def fixed_iteration(x_input, y_target, eta, steps):
	"""
	Return weights calculated from 'steps' number of steps of gradient descent.
	Arguments:
		x_input - *NOT-preprocessed* an array
		y_target - *NOT-preprocessed* a vector of length n
		eta - a float, positve, close to 0
		steps - an int
	"""
	x, y, w = prepare_data(x_input, y_target)
	for _ in range(steps):
		w = update_w(x, y, w, eta)
	return w

x = np.array([[22,7.25],[38,71.2833],[26,7.925],[35,53.1]])
y = np.array([-1,1,1,1])
eta = .1
steps = 100

print(fixed_iteration(x,y, eta, steps))
#--> np.array([-0.9742495,  -0.41389924, 6.8199374 ])
	

def predict(x_input, weights):
	"""
	Return the label prediction, 1 or -1 (an integer), for the given x_input and LR weights.
	Arguments:
		x_input - *NOT-preprocessed* a vector of length d-1
		weights - a vector of length d
	"""
	# begin matrix with ones
	def _prependOnes(mat):
		toStack = [np.ones(1)]
		toStack.append(mat)
		return np.hstack(toStack)
	x_input = _prependOnes(x_input)
	factor = x_input.T@weights
	if factor > 0:
		return 1
	return -1

Xs = np.array([[22,7.25],[38,71.2833],[26,7.925],[35,53.1]])
weights = np.array([0,1,-1])

for X in Xs:
	# print(predict(X,weights))
	pass
	#--> i      1
				# -1
				#  1
				# -1