from scipy.stats import multivariate_normal
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def exploreData():
	# File Paths
	pth = './rsrc'
	mv_path = pth + "/mv.csv"
	unif_path = pth + "/unif.csv"
	mv2_path = pth + "/mv2.csv"
	mv3_path = pth + "/mv3.csv"

	# Read in Data
	mv_df = pd.read_csv(mv_path, index_col = 0)
	unif_df = pd.read_csv(unif_path, index_col = 0)
	mv2_df = pd.read_csv(mv2_path, index_col = 0)
	mv3_df = pd.read_csv(mv3_path, index_col = 0)

	# Create Figure
	fig, (axs) = plt.subplots(2,2, figsize = (6,6))

	# Plot each group in each dataset as unique olor
	for ax, df in zip(axs.flatten(), [mv_df, unif_df, mv2_df, mv3_df]):
		for cat, col in zip(df['cat'].unique(), ["#1b9e77", "#d95f02", "#7570b3"]):
			ax.scatter(df[df.cat == cat].x, df[df.cat == cat].y, c = col, label = cat, alpha = .15)
		ax.legend()

# exploreData()

# Implement K-means ++ to initialize centers
def pick_cluster_centers(points, num_clusters = 3):
	# Create List to store clusters
	clusters = []
	
	# Save list of cluster indicies
	arr_idx = np.arange(len(points))
	
	# Choose first cluster; append to list
	clusters.append( points[np.random.choice(arr_idx)])
	
	# Define function to calculate squared distance
	def dist_sq(x): return np.linalg.norm(x)**2
	
	c_dist = None

	# Add Clusters until reaching "num_clusters"
	while len(clusters) < num_clusters:
		# Calculate distance between latest cluster and rest of points
		new_dist = np.apply_along_axis(np.linalg.norm, 1, points - clusters[-1]).reshape(-1,1)
		
		# Add to distance array - First check to see if distance matrix exists
		if type(c_dist) == type(None):
			c_dist = new_dist
				
		else:
			c_dist = np.concatenate([c_dist, new_dist], axis = 1)
		
		# Calculate probability by finding shortest distance, then normalizing
		c_prob = np.apply_along_axis(np.min, 1, c_dist)
		c_prob = c_prob / c_prob.sum()

		# Draw new cluster according to probability
		clusters.append(points[np.random.choice(arr_idx, p = c_prob)])
					
	return np.array(clusters)

# Function to add pi and Sigma for GMM clusters
def build_GMM_clusters(clusters):
	return [(c, 1/len(clusters), np.array([[1,0],[0,1]])) for c in clusters]


### Question 1
'''
Assign points to clusters according to the k-means algorithm
c_i = argmin_k ||x_i - u_k||**2
	i: point
	c_i: cluster indicator (a cluster)
'''

def assign_clusters_k_means(points, clusters):
	"""
	Determine the nearest cluster to each point, returning an array indicating the closest cluster
	Positional Arguments:
		points: a 2-d numpy array where each row is a different point, and each
			column indicates the location of that point in that dimension
		clusters: a 2-d numpy array where each row is a different centroid cluster;
			each column indicates the location of that centroid in that dimension
	"""
	# NB: "cluster_weights" is used as a common term between functions
	# the name makes more sense in soft-clustering contexts
	dists = [np.apply_along_axis(np.linalg.norm, 1, points-c).reshape(-1,1) for c in clusters]
	dists = np.concatenate(dists, axis=1)
	def get_minimum(dst):
		'''
		takes in a list of the distances from single point to each centroid
		'''
		mn = min(dst)
		return [1 if d==mn else 0 for d in dst]
	return np.apply_along_axis(get_minimum, 1, dists)


points = np.array([[0,1], [2,2], [5,4], [3,6], [4,2]])
clusters = np.array([[0,1],[5,4]])

plt.scatter(points[:, 0], points[:, 1], label='data')
plt.scatter(clusters[:, 0], clusters[:, 1], label='centroids')
plt.legend()

# display the clustering weights
# print(assign_clusters_k_means(points, clusters))
#--> np.array([[1, 0], [1, 0], [0, 1], [0, 1], [0, 1]])

def assign_clusters_soft_k_means(points, clusters, beta):
	"""
	Return an array indicating the porportion of the point
			belonging to each cluster
	Positional Arguments:
			points: a 2-d numpy array where each row is a different point, and each
					column indicates the location of that point in that dimension
			clusters: a 2-d numpy array where each row is a different centroid cluster;
					each column indicates the location of that centroid in that dimension
			beta: a number indicating what distance can be considered "close"
	"""
	def get_distance(diff): return np.exp((-1/beta)*np.linalg.norm(diff))
	dists = [np.apply_along_axis(get_distance, 1, points-c).reshape(-1,1) for c in clusters]
	dists = np.concatenate(dists, axis=1)
	def porportional_wts(dst): return dst/sum(dst)
	return np.apply_along_axis(porportional_wts, 1, dists)


points = np.array([[0,1], [2,2], [5,4], [3,6], [4,2]])
clusters = np.array([[0,1],[5,4]])
beta = 1
cluster_weights = assign_clusters_soft_k_means(points, clusters, beta)

# print(cluster_weights) #--> np.array(  [[0.99707331, 0.00292669],
																			# [0.79729666, 0.20270334],
																			# [0.00292669, 0.99707331],
																			# [0.04731194, 0.95268806],
																			# [0.1315826 , 0.8684174 ]])

'''
phi(k) = (pi_k * N(x_i|u_k, Sigma_k)) / (Sigma_j * pi_j * N(x_i|u_j, Sigma_k))
	phi: the proportion of a point that belongs to a cluster
	i: a point
	k: a cluster
	u: 1d numpy array, centroid location
	pi: numeric, weight of the cluster
	Sigma: 2d numpy array, the cluster's covariance
'''

def assign_clusters_GMM(points, clusters):
	"""    
	Return an array indicating the porportion of the point
		belonging to each cluster
	Positional Arguments:
		points: a 2-d numpy array where each row is a different point, and each
			column indicates the location of that point in that dimension
		clusters: a list of tuples. Each tuple describes a cluster.
			The first element of the tuple is a 1-d numpy array indicating the
				location of that centroid in each dimension
			The second element of the tuple is a number, indicating the weight (pi)
				of that cluster
			The thrid element is a 2-d numpy array corresponding to that cluster's
				covariance matrix.
	"""
	cluster_weights = []
	for p in points:
		# print(p)
		weights = []
		for c in clusters:
			u, pi, Sigma = c
			# print(u, pi, Sigma)
			weight = pi * multivariate_normal(u, Sigma).pdf(p)
			weights.append(weight)
		cluster_weights.append(weights)
	cluster_weights = [wt/sum(wt) for wt in cluster_weights]
	return np.array(cluster_weights)


points = np.array([[0,1], [2,2], [5,4], [3,6], [4,2]])
clusters = [(np.array([0,1]), 1, np.array([[1,0],[0,1]])),
						(np.array([5,4]), 1, np.array([[1,0],[0,1]]))]


# print(assign_clusters_GMM(points, clusters))
#--> np.array([[9.99999959e-01 4.13993755e-08]
						 # [9.82013790e-01 1.79862100e-02]
						 # [4.13993755e-08 9.99999959e-01]
						 # [2.26032430e-06 9.99997740e-01]
						 # [2.47262316e-03 9.97527377e-01]])


'''
u_k = Sum_i(x_i) {given x_i = k} / Sum_i(1) {given x_i = k}
	u_k: centroid for cluster k
	Sum_i: summation across all points
	x_i: a point
	c_i: the cluster point i was assigned to
'''

def update_clusters_k_means(points, cluster_weights):
	"""
	Update the cluster centroids via the k-means algorithm
	Positional Arguments -- 
			points: a 2-d numpy array where each row is a different point, and each
					column indicates the location of that point in that dimension
			cluster_weights: a 2-d numy array where each row corresponds to each row in "points"
					and the columns indicate which cluster the point "belongs" to - a "1" in the kth
					column indicates belonging to the kth cluster
	"""
	cl_points = {i: [] for i, _ in enumerate(cluster_weights[0])}
	# group the points by their clusters
	for p, c in zip(points, cluster_weights):
		idx = np.where(c==1)[0][0]
		cl_points[idx].append(p)
	# reformat by stacking the arrays
	for k in cl_points.keys(): cl_points[k] = np.vstack(cl_points[k])
	# get the averages
	def fun(pts): return np.mean(pts)
	avgs = [np.apply_along_axis(fun, 0, cl_points[k]) for k in cl_points.keys()]
	new_clusts = np.vstack(avgs)
	return new_clusts

points = np.array([[0,1], [2,2], [5,4], [3,6], [4,2]])
cluster_weights = np.array([[1, 0],[1, 0],[0, 1],[0, 1],[0, 1]])

# print(update_clusters_k_means(points, cluster_weights))
#--> np.array([[1. , 1.5], [4. , 4. ]])

'''
u_k = Sum_i(x_i * phi_i(k)) / Sum_i(phi_i(k))
	Sum_i: summation across all points
	x_i: a point
	phi_i(k): weights corresponding to the k clusters
'''

def update_clusters_soft_k_means(points, cluster_weights):
	"""
	Update cluster centroids according to the soft k-means algorithm
	Positional Arguments --
			points: a 2-d numpy array where each row is a different point, and each
					column indicates the location of that point in that dimension
			cluster_weights: a 2-d numpy array where each row corresponds to each row in 
					"points". the values in that row corresponding to the amount that point is associated
					with each cluster.
	"""
	clst_sums = np.apply_along_axis(np.sum, 0, cluster_weights)
	wtd_pts = [np.apply_along_axis(np.sum, 0, points*c.reshape(-1, 1)) \
		for c in cluster_weights.T]
	wtd_pts = np.vstack(wtd_pts)
	new_cents = [w/s for w, s in zip(wtd_pts, clst_sums)]
	new_cents = np.vstack(new_cents)
	return new_cents


points = np.array([[0,1], [2,2], [5,4], [3,6], [4,2]])

cluster_weights= np.array([[0.99707331, 0.00292669],
													 [0.79729666, 0.20270334],
													 [0.00292669, 0.99707331],
													 [0.04731194, 0.95268806],
													 [0.1315826 , 0.8684174 ]])

# print(update_clusters_soft_k_means(points, cluster_weights))
#--> np. array([[1.15246591, 1.59418291],
							# [3.87673553, 3.91876291]])


'''
n_k = Sum(phi_i(k))

pi_k = n_k / n
	Sum: summation at index, i, from 1 to n
	phi_i(k): weights for the current point belongin to one cluster

u_k = (1/n_k) * Sum(phi_i(k) * x_i)
	Sum: summation at index, i, from 1 to n

Sigma_k = (1/n_k) * Sum(phi_i(k) * (x_i - u_k) * (x_i - u_k).T)
'''

def update_clusters_GMM(points, cluster_weights):
	"""
	Update cluster centroids (mu, pi, and Sigma) according to GMM formulas
	Positional Arguments --
			points: a 2-d numpy array where each row is a different point, and each
					column indicates the location of that point in that dimension
			cluster_weights: a 2-d numpy array where each row corresponds to each row in 
					"points". the values in that row correspond to the amount that point is associated
					with each cluster.
	"""
	# general calculations
	n_pts = len(points)
	n_k = np.apply_along_axis(np.sum, 0, cluster_weights)
	# calculate pi
	pi_k = n_k/n_pts
	# calculate mu
	sum_u = [np.apply_along_axis(np.sum, 0, points*cl_wts.reshape(-1, 1)) \
		for cl_wts in cluster_weights.T]
	sum_u = np.vstack(sum_u)
	u_k = [u*n for u, n in zip(sum_u, 1/n_k)]
	u_k = np.vstack(u_k)
	# Calculate sigma
	diffs = [points - u for u in u_k]
	def mult(diff):
		return diff.reshape(-1, 1)@diff.reshape(1, -1)
	diff_mult = [np.apply_along_axis(mult, 1, df) for df in diffs]
	in_sum = [[cl_wt*dm \
		for cl_wt, dm in zip(cl_wts, dms)] \
			for cl_wts, dms in zip(cluster_weights.T, diff_mult)]
	summation = [sum(i_sm) for i_sm in in_sum]
	z_k = [s/n for s, n in zip(summation, n_k)]
	# group into cluster based list
	new_clusts = [(u, p, z) for u, p, z in zip(u_k, pi_k, z_k)]
	return new_clusts



points = np.array([[0,1], [2,2], [5,4], [3,6], [4,2]])
cluster_weights = np.array([[9.99999959e-01, 4.13993755e-08],
														[9.82013790e-01, 1.79862100e-02],
														[4.13993755e-08, 9.99999959e-01],
														[2.26032430e-06, 9.99997740e-01],
														[2.47262316e-03, 9.97527377e-01]])

new_clusters = update_clusters_GMM(points, cluster_weights)

# print(new_clusters)
for t in new_clusters:
	print(t)
'''
[(array([0.99467691, 1.49609648]), #----> mu, centroid 1
0.3968977347767351, #-------------------> pi, centroid 1
array([[1.00994319, 0.50123508],
				[0.50123508, 0.25000767]])), #---> Sigma, centroid 1

(array([3.98807155, 3.98970927]), #----> mu, centroid 2
0.603102265479875, #-------------------> pi, centroid 2
array([[ 0.68695286, -0.63950027], #---> Sigma centroid 2
				[-0.63950027,  2.67341935]]))]
'''
        

# plt.show()