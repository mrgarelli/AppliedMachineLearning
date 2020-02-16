import numpy as np
import matplotlib.pyplot as plt

def assign_clusters_k_means(points, clusters):
	# print()
	# print('difference')
	# print([points-c for c in clusters])
	dists_to_clust = np.concatenate(
		[np.apply_along_axis(np.linalg.norm, 1, points-c).reshape(-1,1) for c in clusters],
		axis = 1
		)

	# Function to convert minimum distance to 1 and others to 0
	def find_min(x):
		m = np.min(x)
		return [1 if n == m else 0 for n in x]

	# Apply function
	# print()
	# print('min')
	cluster_assignments = np.apply_along_axis(find_min, 1, dists_to_clust)
	return cluster_assignments

points = np.array([[0,1], [2,2], [5,4], [3,6], [4,2]])
clusters = np.array([[0,1],[5,4]])

# display the clustering weights

# print()
# print('Answer')
# print(assign_clusters_k_means(points, clusters))
#--> np.array([[1, 0], [1, 0], [0, 1], [0, 1], [0, 1]])


def update_clusters_GMM(points, cluster_weights):
	# general calculations
	n_pts = len(points)
	n_k = np.apply_along_axis(np.sum, 0, cluster_weights)

	# calculate pi
	pi_k = n_k/n_pts
	# print(pi_k.reshape(-1, 1))
	# print()

	# calculate mu
	sum_u = [np.apply_along_axis(np.sum, 0, points*cl_wts.reshape(-1, 1)) \
		for cl_wts in cluster_weights.T]
	sum_u = np.vstack(sum_u)
	u_k = [u*n for u, n in zip(sum_u, 1/n_k)]
	u_k = np.vstack(u_k)
	# print(u_k)
	# print()

	for ci, c in enumerate(cluster_weights.T):
		for cw, p in zip(c, points):
			diff = p - u_k[ci]
			# print(diff)
			diff_mult = diff.reshape(-1, 1)@diff.reshape(1, -1)
			# print(diff_mult)
			# print()
			in_sum = cw * diff_mult
			print(in_sum)
		print()

	new_clusts = []
	return new_clusts


points = np.array([[0,1], [2,2], [5,4], [3,6], [4,2]])
cluster_weights = np.array([[9.99999959e-01, 4.13993755e-08],
														[9.82013790e-01, 1.79862100e-02],
														[4.13993755e-08, 9.99999959e-01],
														[2.26032430e-06, 9.99997740e-01],
														[2.47262316e-03, 9.97527377e-01]])

new_clusters = update_clusters_GMM(points, cluster_weights)