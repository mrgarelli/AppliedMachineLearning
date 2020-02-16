import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import regression

plt.rcParams['figure.figsize'] = (20.0, 10.0)
### Read in the data
tr_path = '../resource/train.csv'
test_path = '../resource/test.csv'
data = pd.read_csv(tr_path)
### The .head() function shows the first few lines of data for perspecitve

# print(data.head())

### Lists column names

# print(data.columns)

### number columns are in `data`?
# print(len(data.columns))

### We can plot the data as follows
### Price v. living area
### with matplotlib
Y = data['SalePrice']
X = data['GrLivArea']

# plt.scatter(X, Y, marker = "x")

### Annotations
# plt.title("Sales Price vs. Living Area (excl. basement)")
# plt.xlabel("GrLivArea")
# plt.ylabel("SalePrice")
### price v. year
### Using Pandas



# understand matrix inverse: https://www.mathsisfun.com/algebra/matrix-inverse.html

# print("test",inverse_of_matrix([[1,2],[3,4]]), "\n")

# print("From Data:\n", inverse_of_matrix(data.iloc[:2,:2]))
#
# AUTOGRADER TEST - DO NOT REMOVE
#

data = regression.read_to_df(tr_path)
small = data.head()
print(small)

# print(regression.select_columns(small, ['YrSold', 'SaleType']))
# print(list(small.index))

# data.plot('YearBuilt', 'SalePrice', kind = 'scatter', marker = 'x')
data.plot.scatter(x='SalePrice', y='YrSold', marker='o')
# plt.show()

newFrame = regression.column_cutoff(small, [('MSSubClass', 20, 60)])
print(newFrame)

#
# AUTOGRADER TEST - DO NOT REMOVE
#
