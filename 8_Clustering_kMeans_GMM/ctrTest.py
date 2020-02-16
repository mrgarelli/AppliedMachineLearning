import numpy as np

dists = [1, 2, 2, 1, 5, 6, 6, 3]
print(np.min(dists))
minm = min(dists)
indices = [i for i, dist in enumerate(dists) if dist==minm]

print(indices)
