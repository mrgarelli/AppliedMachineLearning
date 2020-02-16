import numpy as np

def pred_from_split(X, y, col_idx, split_value):
	def node_classes(col, split_value, labels):
		# subset labels by observations that are <= / > the split value
		le_node = labels[col <= split_value]
		g_node = labels[col > split_value]
		# count members of each class at each node # c1 corresponds to "1's"
		# c2 corresponds to "0's
		le_c1 = np.count_nonzero(le_node)
		le_c2 = len(le_node) - le_c1
		g_c1 = np.count_nonzero(g_node)
		g_c2 = len(g_node) - g_c1
		return le_c1, le_c2, g_c1, g_c2

	# Return count of each class at each node using helper function
	le_c1, le_c2, g_c1, g_c2 = node_classes(X[:,col_idx], split_value, y)

	def pred_for_node(cl1, cl2):
		if cl1 > cl2: return True
		elif cl1 < cl2: return False
		else: return None

	# Create intial predictions
	left = pred_for_node(le_c1,le_c2)
	right = pred_for_node(g_c1, g_c2)

	# Check to see if one of the predictions came back as "None"
	if ((right == None) or (left == None)) and (right != left):
		if left == None: left = not right
		else: right = not left


	# Check to see if both predictions came back as "None"
	if (right == None) and (left == None):
		right = True; left = True

	return (int(left), int(right))

X = np.array([[0.5, 3. ], [1.,  2. ], [3.,  0.5],
				  [2.,  3. ], [3.,  4. ]])
y = np.array([ 1, 1, 0, 0, 1])
col_idx = 0
split_value = 1.5
pred_at_nodes = pred_from_split(X, y, col_idx, split_value)
print(pred_at_nodes) # --> (1, 0)