import pandas as pd

one = [12254, 12065, 10906, 9046, 23956]
three = [11157, 15098, 7809, 6636, 13355]
total = [22550, 19497, 15188, 13019, 9992]
cols = ['one', 'three', 'thirteen']
trials = [1, 2, 3, 4, 5]

df = pd.DataFrame(index=trials, columns=cols)
df.one = one
df.three = three
df.thirteen = total
print(df.head())
print() 

def drop_val(df, to_drop):
	"""
	Drop rows from the DataFrame containing the specified values
	"""
	def remove_rows_4_col(col):
		idxs = df[df[col] == to_drop].index
		df.drop(idxs, inplace=True)
	for col in df.columns: remove_rows_4_col(col)
	return df

smaller_df = drop_val(df, 10906)

print()
# print(smaller_df)
