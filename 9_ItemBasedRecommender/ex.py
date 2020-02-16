import pandas as pd
import numpy as np

cols = ['Mozart', 'Bach', 'Chopin', 'Brahms', 'Wagner', 'Liszt']
rows = ['Abel', 'Baker', 'Charlie', 'David', 'Erik', 'Frank']

df = pd.DataFrame(index=rows, columns=cols)

Mozart = [0, 5, 5, 5, 3, 2]
Bach = [1, 0, 4, 5, 3, 2]
Chopin = [2, 3, 0, 5, 4, 1]
Brahms = [2, 2, 5, 0, 3, 1]
Wagner = [1, 3, 3, 2, 0, 3]
Liszt = [3, 3, 2, 1, 2, 0]

df.Mozart = Mozart
df.Bach = Bach
df.Chopin = Chopin
df.Brahms = Brahms
df.Wagner = Wagner
df.Liszt = Liszt

print(df.head())


### Defining function for pearson's Correlation Coefficient
def p_sim(ser1, ser2):
	def normalize(raw):
			return .5 + (raw/2)
	corr = np.corrcoef(ser1, ser2)[0][1] ### returns 2x2 array with correlation to self(1) on diagonal
	return normalize(corr)

### Defining drop_rows_with_zeros
def drop_rows_with_zeros(df):
	def nonzero(ser): return ser.to_numpy().nonzero()
	keep = np.intersect1d(nonzero(df.iloc[:,0]), nonzero(df.iloc[:,1]))
	return df.iloc[keep,:]

'''
calculates mozarts similarity to every other artist (according to all evaluators)
'''
mozSimScores = {}
for mus in df.columns[1:]: # for each musician that isn't mozart
	no_zeros = drop_rows_with_zeros(df[['Mozart', mus]])
	# print()
	# print(no_zeros)
	sim_corr_coeff = round(p_sim(no_zeros.iloc[:,0], no_zeros.iloc[:,1]),2)
	# print()
	# print(sim_corr_coeff)
	mozSimScores[mus] = sim_corr_coeff

print()
print(mozSimScores)