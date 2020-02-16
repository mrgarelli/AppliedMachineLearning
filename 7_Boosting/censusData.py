import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

col_names = [
"age", "workclass", "fnlwgt", "education",
"education-num", "marital-status", "occupation", "relationship",
"race", "sex", "capital-gain", "capital-loss", "hours-per-week",
"native-country", "income"
]

data_path = "./rsrc/adult.data"

data = pd.read_csv(data_path, header = None, names = col_names)
# print(data.head())

cols = ["age", "workclass", "education-num", "occupation", "sex", "hours-per-week", "income"]
data = data[cols] # select specific columns
# print(data.head())

print(data.describe(include='all'))

def showPlots():
	plt.figure(figsize=(10,10))
	for i, col in enumerate(data.drop("hours-per-week", axis = 1)):
			d = data[col].value_counts().sort_index()
			plt.bar(d.index, d)
			plt.xticks(rotation = 90)
			plt.title(col)
			plt.show()

	plt.hist(data['hours-per-week'], bins = 20);
	plt.title("Hours Per Week");

# showPlots()

from sklearn.preprocessing import OneHotEncoder

ex_df = pd.DataFrame(
	np.array([['b', 'a', 'c', 'a', 'c',], ["z","y","y","y","z"]]).T,
	columns = ["beg","end"],
	index = range(500,505))

test_df = pd.DataFrame(
	np.array([["c","b"],["y","y"]]).T,
	columns = ["beg","end"],
	index = [56,72])

print("Initial DataFrame:")
print(ex_df)

print("\nTest df")
print(test_df)

# Instantiate OneHotEncoder
# sparse = False means data will not be stored in sparse matrix
ohe = OneHotEncoder(sparse = False)

# Fitting OHE with the "training" data
ohe.fit(ex_df)

# Transforming the "training" dat
tr_vals = ohe.transform(ex_df)

print("\nTransformed values")
print(tr_vals)

print("\nCategories")
print(ohe.categories_)

# Creating column names from `.categories_`
ohe_cats = np.concatenate(ohe.categories_)

# In creation of new df. Note the use of np.concatenate
final_df = pd.DataFrame(tr_vals, columns = ohe_cats)

print("\nFinal DataFrame")
print(final_df)

# Putting everything together to transform test data
print("\nTransformed test df")
print(pd.DataFrame(ohe.transform(test_df), columns= ohe_cats))