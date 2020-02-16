
import numpy as np
import pandas as pd
from collections import namedtuple
from itertools import permutations, chain, product
from sklearn.cluster import KMeans
from inspect import Signature, Parameter


# Define path constants
ROOT_DIR = '../resource/assignment11/'
CORN_2013_2017 = 'corn2013-2017.txt'
CORN_2015_2017 = 'corn2015-2017.txt'
OHL = 'corn_OHLC2013-2017.txt'

corn13_17 = pd.read_csv(ROOT_DIR+CORN_2013_2017, names = ("week","price") )
corn15_17 = pd.read_csv(ROOT_DIR+CORN_2015_2017, names = ("week","price"))
OHL_df = pd.read_csv(ROOT_DIR+OHL, names = ("week","open","high","low","close"))

# print(corn13_17.head())
# print(corn13_17.info())
# print()

'''
Question 2
Create a function called `generate_cluster_assignments`
The function should take 2 parameters: 
	1) pandas Series
	2) number of clusters

Instantiate a sklearn KMeans class with specified number of clusters and random_state=24.
Return a pandas Series of cluster labels for each observation in the sequence.
A KMeans object can be instantiated via: clusterer = KMeans(args)
	That KMeans object has `.fit()` and `.predict()` methods

NOTE: Your particular labels might not match exactly, but the CLUSTERS should be the same
'''

def generate_cluster_assignments(ser, clusters):
	mod = KMeans(clusters, random_state=24)
	df = pd.DataFrame(ser)
	mod.fit(df)
	return mod.predict(df)

data_series = pd.Series([1,2,3,2,1,2,3,2,1,2,3,2,1,2,3,2,1,6,7,
												 8,7,6,7,8,6,7,6,7,8,7,7,8,56,57,58,59,57,58,6,7,8,1,2])
# labels = generate_cluster_assignments(data_series, clusters = 3)
# print(labels)

# labels --> array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
#                   2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 2, 1, 1])

# Cluster 2013-2017
corn13_17_seq = generate_cluster_assignments(corn13_17[['price']], 5)
print(corn13_17_seq)

# Almost all functions require this constant as an argument
STATE_LIST = ['S1', 'S2']

# Initialze state transition probabilities (2 states)
STATE_TRANS_PROBS = [0.4, 0.6, 0.35, 0.55]