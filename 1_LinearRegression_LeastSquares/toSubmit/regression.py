### This cell imports the necessary modules and sets a few plotting parameters for display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)

# understand matrix inverse: https://www.mathsisfun.com/algebra/matrix-inverse.html
def inverse_of_matrix(mat):
    """Calculate and return the multiplicative inverse of a matrix.
    
    Positional argument:
        mat -- a square matrix to invert
    
    Example:
        sample_matrix = [[1, 2], [3, 4]]
        the_inverse = inverse_of_matrix(sample_matrix)
        
    Requirements:
        This function depends on the numpy function `numpy.linalg.inv`. 
    """
    matrix_inverse = np.linalg.inv(mat)
    return matrix_inverse


import pandas as pd
def read_to_df(file_path):
	"""Read on-disk data and return a dataframe."""

	data = pd.read_csv(file_path)
	
	return data


def select_columns(data_frame, column_names):
    """Return a subset of a data frame by column names.

    Positional arguments:
        data_frame -- a pandas DataFrame object
        column_names -- a list of column names to select

    Example:
        data = read_to_df('train.csv')
        selected_columns = ['SalePrice', 'GrLivArea', 'YearBuilt']
        sub_df = select_columns(data, selected_columns)
    """

    return data_frame[column_names]


def column_cutoff(data_frame, cutoffs):
	"""Subset data frame by cutting off limits on column values.
	
	Positional arguments:
		data -- pandas DataFrame object
		cutoffs -- list of tuples in the format: 
		(column_name, min_value, max_value)
		
	Example:
		data_frame = read_into_data_frame('train.csv')
		# Remove data points with SalePrice < $50,000
		# Remove data points with GrLiveAre > 4,000 square feet
		cutoffs = [('SalePrice', 50000, 1e10), ('GrLivArea', 0, 4000)]
		selected_data = column_cutoff(data_frame, cutoffs)
	"""
	newFrame = data_frame
	for (column_name, min_value, max_value) in cutoffs:
		newFrame = newFrame[newFrame[column_name] <= max_value]
		newFrame = newFrame[newFrame[column_name] >= min_value]
	return newFrame


### GRADED
### Build a function  called "least_squares_weights"
### take as input two matricies corresponding to the X inputs and y target
### assume the matricies are of the correct dimensions

### Step 1: ensure that the number of rows of each matrix is greater than or equal to the number
### of columns.
### ### If not, transpose the matricies.
### In particular, the y input should end up as a n-by-1 matrix, and the x input as a n-by-p matrix

### Step 2: *prepend* an n-by-1 column of ones to the input_x matrix

### Step 3: Use the above equation to calculate the least squares weights.

### NB: `.shape`, `np.matmul`, `np.linalg.inv`, `np.ones` and `np.transpose` will be valuable.
### If those above functions are used, the weights should be accessable as below:  
### weights = least_squares_weights(train_x, train_y)
### weight1 = weights[0][0]; weight2 = weights[1][0];... weight<n+1> = weights[n][0]

### YOUR ANSWER BELOW


def least_squares_weights(input_x, target_y):
    """Calculate linear regression least squares weights.
    
    Positional arguments:
        input_x -- matrix of training input data
        target_y -- vector of training output values
        
        The dimensions of X and y will be either p-by-n and 1-by-n
        Or n-by-p and n-by-1
        
    Example:
        import numpy as np
        training_y = np.array([[208500, 181500, 223500, 
                                140000, 250000, 143000, 
                                307000, 200000, 129900, 
                                118000]])
        training_x = np.array([[1710, 1262, 1786, 
                                1717, 2198, 1362, 
                                1694, 2090, 1774, 
                                1077], 
                               [2003, 1976, 2001, 
                                1915, 2000, 1993, 
                                2004, 1973, 1931, 
                                1939]])
        weights = least_squares_weights(training_x, training_y)
        
        print(weights)  #--> np.array([[-2.29223802e+06],
                           [ 5.92536529e+01],
                           [ 1.20780450e+03]])
                           
        print(weights[1][0]) #--> 59.25365290008861
        
    Assumptions:
        -- target_y is a vector whose length is the same as the
        number of observations in training_x
    """
    

    return ''

### GRADED
### Why, in the function  above, is it necessary to prepend a column of ones
### 'a') To re-shape the matrix
### 'b') To create an intercept term
### 'c') It isn't needed, it's just meant to be confusing
### 'd') As a way to make sure the weights turn out positive
### Assign the character asociated with your choice as a string to ans1
### YOUR ANSWER BELOW

ans1 = ''
#
# AUTOGRADER TEST - DO NOT REMOVE
#