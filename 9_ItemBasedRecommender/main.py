import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

def euclidean_distance(one, two):
	return np.sqrt(np.sum((one - two)**2))

def similarity_score(euc_dist):
	return 1/(1 + euc_dist)

# sum(A_c * simScore(p, c) / sum(simScore(p, c))
# 	sum: sum for each column, c (except the one being predicted)
# 	p: denotes the name of the column we are trying to predict
# 	A_c: how our critic ranked all the other musicians
def predict(rank, sim_scores):
	'''
	Inputs:
		rank: a weighted rating based on the bias of the rater
			our evaluators rating for all other artists
			this discludes the artist's score we are predicting our evaluator will choose
		sim_scores: similarity scores
			between artist to predict and all other artists (according to every evaluator)
	return a prediction
	'''
	prod = rank * sim_scores
	prediction = sum(prod)/sum(sim_scores)
	return prediction

brahms = np.array([2, 2, 5])
wagner = np.array([1, 1, 3])
euc_dist = np.sqrt(np.sum((brahms - wagner)**2))
ss = similarity_score(euc_dist)
# print(ss)


simScore = {'Bach': 0.97, 'Chopin': 0.84, 'Brahms': 0.81, 'Wagner': 0.33, 'Liszt': 0.5}
sims = np.fromiter(simScore.values(), dtype=float)
rank = np.array([1, 2, 2, 1, 3])

prediction = predict(rank, sims)
print(prediction)

### Question 7
'''
consider 3 products: A, B, and X
userN has ranked products A, and B, giving each a '5' and we are predicting their score for X
user1 has ranked products A, and X, giving each a "3"
user2 has ranked products B and X, giving each a '2'

Assume this is all the data we have for a product.
What will be the similarity score between prodA and prodX? assign number to ans1
What will be the similarity score between prodB and prodX? assign number to ans2
YOUR ANSWER BELOW
'''
userN = np.array([5, 5]) # ranked A and B, predicting their score for X
user1 = np.array([3, 3]) # ranked A and X
user2 = np.array([2, 2]) # ranked B and X

sim_scores = np.array([
	similarity_score(euclidean_distance(3, 3)), 
	similarity_score(euclidean_distance(2, 2))
	])

rankings = np.array([5, 5])
print(predict(rankings, sim_scores))