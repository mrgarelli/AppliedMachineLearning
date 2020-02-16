import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

plt.style.use('dark_background')

FEATURE_NAMES = '../resource/features.txt'
TRAIN_DATA = '../resource/train/X_train.txt'
TRAIN_LABELS = '../resource/train/y_train.txt'

# Ensure Data is consistent

# read feature names
feats = pd.read_csv(FEATURE_NAMES, sep='\n', header=None)

# read in training data
har_train = pd.read_csv(TRAIN_DATA, sep='\s+', header=None)

# read in training labels, and clean them.
har_train_labels = pd.read_csv(TRAIN_LABELS, sep='\n', header=None)
clean_features = [feat[0].split(' ')[1] for feat in feats.values]
har_train.columns = clean_features

har_train_labels = pd.read_csv(TRAIN_LABELS, sep='\n', header=None)
har_train_labels.columns = ['label']
y = har_train_labels.loc[:, 'label']


def run_and_test_knn_model():
	### This cell creates X_train3, X_test3, y_train3, and y_test3; used below.
	X_train3, X_test3, y_train3, y_test3 = train_test_split(
		har_train,
		y,
		test_size = .4,
		random_state = 2001
		)

	# Instantiate classifier
	# NB: Default distance is Euclidean
	knn = KNeighborsClassifier(n_neighbors = 10)

	# Fit model with training data
	knn.fit(X_train3, y_train3)

	# Create predictions for first 200 test observations
	# # (As was done above with customKNN)
	skpreds = knn.predict(X_test3)

	print("sklearn prediction performance")
	print(classification_report(y_test3, skpreds))

# run_and_test_knn_model()

y = har_train_labels 
X = har_train
X_train, X_test, y_train, y_test = train_test_split(
	X,
	y,
	test_size=.3,
	random_state=24
	)

from sklearn.metrics import recall_score
def score_multiple_knn_attempts():
	### Calculating Recall scores for multiple "n-neighbors"
	recall_scores = {}
	for n in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,50,75,100]:
		knn = KNeighborsClassifier(n_neighbors=n)
		knn.fit(X_train, y_train)
		recall_scores[n] = recall_score(y_test, knn.predict(X_test), average = None)

	### Put recall scores into DataFrame
	scores_df = pd.DataFrame(recall_scores).T
	scores_df.columns = [str(i) for i in range(1,7)]
	scores_df.index = scores_df.index.astype(str)

	### Create plot of recall scores
	plt.figure(figsize = (10,10))
	for col in scores_df:
		if col != 'n_neighbors':
			plt.plot(scores_df[col], label = col)
		
	plt.ylabel(" Recall Score", fontsize = 12)
	plt.xlabel("n_neighbors (NB: not an interval scale)", fontsize = 12)
	plt.legend(title = "activity")

# score_multiple_knn_attempts()
# plt.show()