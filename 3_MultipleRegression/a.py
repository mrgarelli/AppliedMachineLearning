import numpy as np

to_pred2 = np.array([1,1700,1980])

if type(to_pred2[0]) != float():
	to_pred2 = to_pred2.astype(float)

print(to_pred2)