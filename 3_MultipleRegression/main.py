import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tools.Regression import Regression

plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (20.0, 10.0)

### Read in the data
tr_path = '../resource/train.csv'
df = pd.read_csv(tr_path)

# create a regression object from the df
reg = Regression(df)

# small dataframe for testing
dfSmall = df.head()

# Information Extraction
# print(df.head())
# print(df.columns)
# print(df.describe())
# print(list(dfSmall.index))

# __________________________________________________
# Example of multiple regression - predicting one target from many inputs
# Linear multiple-regression: y = m1*x1 + m2*x2 + b
# avoid non-relevant inputs
# avoid multicollinearity
# https://www.youtube.com/watch?v=AkBjJ6OunR4

reg.choose_params('SalePrice', ['GrLivArea', 'YearBuilt'])
reg.least_squares()

reg.generate_inputs()
reg.predict(reg.x_generated)

# determine quality of fit
reg.root_mean_squared_error()
reg.r_squared()

# check model attributes
# print(reg.y_predicted)
# print(reg.rmse)
# print(reg.rs)

### We can plot the data as follows
### Price v. living area
Y = df['SalePrice']
X = df['GrLivArea']
plt.scatter(X, Y, marker = "x")
plt.plot(reg.x_generated[:, 0], reg.y_predicted[:, 0], label='linear regression')
plt.title("Sales Price vs. Living Area (excl. basement)")
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.legend()
# df.plot('YearBuilt', 'SalePrice', kind = 'scatter', marker = 'x')
# plt.show()
