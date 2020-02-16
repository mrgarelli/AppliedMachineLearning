import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tools.Regression import Regression

plt.style.use('dark_background')

FEATURE_NAMES = '../resource/features.txt'
TRAIN_DATA = '../resource/train/X_train.txt'
TRAIN_LABELS = '../resource/train/y_train.txt'

# read feature names
feats = pd.read_csv(FEATURE_NAMES, sep='\n', header=None)

# read in training data
har_train = pd.read_csv(TRAIN_DATA, sep='\s+', header=None)

# read in training labels
har_train_labels = pd.read_csv(TRAIN_LABELS, sep='\n', header=None, names=["label"], squeeze = True)

# number of rows and columns
# print(har_train.head())
# print(feats.head())

# print(len(har_train.columns))
# print(len(har_train.index))

# change the headers
har_train.columns = feats.iloc[:,0]
# print(har_train.head())

reg = Regression(har_train)
# print(reg.numNulls())

# seaborn
first_twenty = har_train.iloc[:, :20] # pull out first 20 feats
corr = first_twenty.corr()  # compute correlation matrix
mask = np.zeros_like(corr, dtype=np.bool)  # make mask
mask[np.triu_indices_from(mask)] = True  # mask the upper triangle

fig, ax = plt.subplots(figsize=(11, 9))  # create a figure and a subplot
cmap = sns.diverging_palette(220, 10, as_cmap=True)  # custom color map
sns.heatmap(
	corr,
	mask=mask,
	cmap=cmap,
	center=0,
	linewidth=0.5,
	cbar_kws={'shrink': 0.5}
)

# plt.show()

# print(har_train_labels)

# find label classifiers that occur the most/least
activities = {
	1: 0,
	2: 0,
	3: 0,
	4: 0,
	5: 0,
	6: 0,
	}
for lbl in har_train_labels:
	activities[lbl] += 1
vals = activities.values()
# print(min(vals))
# print(max(vals))

# concatenate the target variable
# give target and observations conventional names
y = har_train_labels 
X = har_train

# add labels into data frame
# print(har_train.shape)
data = pd.concat([X, y], axis=1)
# print(data.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=24)


### Please find the Euclidean distance between the points (1,2,3,-4,6) and (10,2,32,-2,0)
def euclidean_distance(one, two):
	sq_diff = [(o - t)**2 for o, t in zip(one, two)]
	return sum(sq_diff)**(.5)

p1 = (1,2,3,-4,6)
p2 = (10,2,32,-2,0)

np_ed = np.linalg.norm(np.array(p1)-np.array(p2))
# print(euclidean_distance(p1, p2))
# print(np_ed)


### ACCEPT two inputs:
### An observation from a data set.  e.g: har_train.iloc[50,:]
### The full data set. e.g. har_train.

### Create a <list> or numpy array of distances between:
### ### that single point, and all points in the full dataset

### RETURN the list of distances SORTED from smallest to largest.

### Notes:
### Use `np.linalg.norm()`, as described in above cell.
### The smallest distance should be 0.
def all_distances(test_point, data_set):
	"""
	Find and return a list of distances between the "test_point"
	and all the points in "data_set", sorted from smallest to largest.
	
	Positional Arguments:
		test_point -- a Pandas Series corresponding to a row in "data_set"
		data_set -- a Pandas DataFrame
	
	"""
	# a dataframe of the differences from the point slice to each verticle element
	diff_df = data_set - test_point
	dists = diff_df.apply(np.linalg.norm, axis='columns')
	return np.sort(dists.values)

test_point = har_train.iloc[50,:]
data_set = har_train

# print(all_distances(test_point, data_set)[:5])
#--> [0.0, 2.7970187358249854, 2.922792670143521, 2.966555149052483, 3.033982453218797]
	


### ACCEPT three inputs:
### 1&2: numpy arrays, corresponding to 1: a numeric column and 2: a label column.
### ### The i-th member of the numeric column corresponds to the i-th member of the label column
### 3: an integer (>0); n.

### RETURN a list (or numpy array) of the n labels corresponding to 
### ### the n smallest values in the numeric column.
### Make sure the order of labels corresponds to the order of values.

### Hint: The labels are found in har_train_labels or y
### Hint: `pd.concat()` might be useful for this or subsequent exercisces  
### YOUR ANSWER BELOW

def labels_of_smallest(numeric, labels, n):
	
	"""
	Return the n labels corresponding to the n smallest values in the "numeric"
	numpy array.
	
	Positional Arguments:
		numeric -- a numpy array of numbers
		labels -- a numpy array of labels (string or numeric)
			corresponding to the values in "numeric"
		n -- a positive integer
		
	"""
	indices = numeric.argsort()
	return labels[indices][:n]

numeric = np.array([7,6,5,4,3,2,1])
labels = np.array(["a","a","b","b","b","a","a"])
n = 6

# print(labels_of_smallest(numeric, labels, n))
#--> np.array(['a', 'a', 'b', 'b', 'b', 'a'])


from collections import Counter
### ACCEPT a non-empty numpy array of labels as input
### RETURN the value that appears most frequently in that array
### In the case of of a tie, RETURN the value in the tie that appears first in the array
def label_voting(labels):
	"""
	Given a numpy array of labels. Return the label that appears most frequently
	If there is a tie for most frequent, return the label that appears first.
	
	Positional Argument:
		labels -- a numpy array of labels
	"""
	mc = Counter(labels).most_common()
	if len(mc) == 1:
		return mc[0][0]
	fKeys = [mc[0][0], mc[1][0]]
	fVals = [mc[0][1], mc[1][1]]
	if fVals[0] > fVals[1]: # we have a winner
		return fKeys[0]
	# otherwise, two occur equally
	for lb in labels:
		if lb in fKeys:
			return lb

lab1 = np.array([1, 2, 2, 3, 3])
lab2 = np.array(["a", "a", "b", "b", "b"])

# print(label_voting(lab1)) #--> 2
# print(label_voting(lab2)) #--> "b"

def custom_KNN( point, X_train, y_train, n):
	"""
	Predict the label for a single point, given training data and a specified
	"n" number of neighbors.
	
	Positional Arguments:
		point -- a pandas Series corresponding to an observation of a point with
				unknown label.
		x_train -- a pandas DataFrame corresponding to the measurements
			of points in a dataset. Assume all values are numeric, and
			observations are in the rows; features in the columns
		y_train -- a pandas Series corresponding to the labels for the observations
			in x_train
	"""

	def all_distances(test_point, data_set):
		"""
		Find and return a list of distances between the "test_point"
		and all the points in "data_set", sorted from smallest to largest.
		
		Positional Arguments:
			test_point -- a Pandas Series corresponding to a row in "data_set"
			data_set -- a Pandas DataFrame
		
		"""
		# a dataframe of the differences from the point slice to each verticle element
		diff_df = data_set - test_point
		dists = diff_df.apply(np.linalg.norm, axis='columns')
		return dists.values

	def labels_of_smallest(numeric, labels, n):
		
		"""
		Return the n labels corresponding to the n smallest values in the "numeric"
		numpy array.
		
		Positional Arguments:
			numeric -- a numpy array of numbers
			labels -- a numpy array of labels (string or numeric)
				corresponding to the values in "numeric"
			n -- a positive integer
			
		"""
		indices = numeric.argsort()
		return labels[indices][:n]


	def label_voting(labels):
		"""
		Given a numpy array of labels. Return the label that appears most frequently
		If there is a tie for most frequent, return the label that appears first.
		
		Positional Argument:
			labels -- a numpy array of labels
		"""
		mc = Counter(labels).most_common()
		if len(mc) == 1:
			return mc[0][0]
		fKeys = [mc[0][0], mc[1][0]]
		fVals = [mc[0][1], mc[1][1]]
		if fVals[0] > fVals[1]: # we have a winner
			return fKeys[0]
		# otherwise, two occur equally
		for lb in labels:
			if lb in fKeys:
				return lb

	# data = pd.concat([X_train, y_train], axis=1)
	dist_lst = all_distances(point, X_train)
	lbls_lst = labels_of_smallest(dist_lst, y_train.values, n)
	lbl = label_voting(lbls_lst)
	return lbl

point = pd.Series([1,2])
X_train = pd.DataFrame([[1,2],[3,4],[5,6]])
y_train = pd.Series(["a","a","b"])
n = 2
# print(X_train)
# print(custom_KNN(point, X_train, y_train, n)) #--> 'a'

# Create New tts
def run_our_custom_knn_classifier():
	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=.3,
		random_state=24
	)

	print("Total 'test' observations:", len(X_test))
	print("Classifying every point in X_test would take too long - classify the first 200")
	custom_preds = []
	for i, idx in enumerate(X_test.index[:200]):
		if i % 100 == 0: print(i)
		pred = custom_KNN(X_test.loc[idx,:], X_train, y_train, 5)
		custom_preds.append(pred)
	return custom_preds

# custom_preds =  run_our_custom_knn_classifier()

# print("\nHome-Built prediction performance")
# print(classification_report(y_test[:200], custom_preds))

def run_sk_learn_k_nearest_neighbors_classifier():
	# Import
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.metrics import classification_report

	# Instantiate classifier
	# NB: Default distance is Euclidean
	knn = KNeighborsClassifier(n_neighbors = 5)

	# Fit model with training data
	knn.fit(X_train, y_train)

	# Create predictions for first 200 test observations
	# # (As was done above with customKNN)
	print(X_test[:200].shape[0])
	skpreds = knn.predict(X_test[:200])

	print("sklearn prediction performance")
	print(classification_report(y_test[:200], skpreds))
	return skpreds

skpreds = run_sk_learn_k_nearest_neighbors_classifier()

### The below lines of code will compare the performance of your home-built classification with
### The sklearn predictions -- if all the cells above were run sucessfully, you should see identical scores

### The below lines of code will explicitly compare predictions:
### "differences" should == 0!

### NB: Commenting/uncommenting multiple lines in Jupyter can be accomplished with:
### <ctrl-/> on windows and <cmd-/> on mac
differences = 0
for cust, sk in zip(custom_preds, skpreds):
	if cust != sk:
		differences +=1
print("Total Differences:", differences)