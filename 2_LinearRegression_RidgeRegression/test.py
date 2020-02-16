import numpy as np

def vertMat(li, prepend=False):
	'''
	returns a 1d vertical matrix from a list (or 1d numpy array)
	prepend option for the 1st column to be ones
	'''
	n = len(li)
	if (prepend == False):
		return np.array(li).reshape((n, 1))
	else:
		toStack = (vertMat(np.ones(n)), vertMat(li))
		return np.hstack(toStack)

ar1 = vertMat([1, 2, 3, 4])
ar2 = vertMat([5, 6, 7, 8])

print(ar1)
toStack = (ar1, ar2)

mat = np.hstack(toStack)
print(mat)

print(mat[:, 1])