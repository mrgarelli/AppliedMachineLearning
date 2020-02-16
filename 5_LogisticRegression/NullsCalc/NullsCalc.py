import pandas as pd
from itertools import compress

class NullsCalc:
	def __init__(self, df):
		self.df = df
		self.numNulls = {}
		self.percentNulls = {}
		self._nullsDict()

	def _nullsDict(self):
		for colName in self.df.columns:
			col = self.df[colName].values
			boolMx = pd.isna(col)
			nulls = sum(boolMx)
			self.numNulls[colName] = nulls
			self.percentNulls[colName] = nulls/len(boolMx)

	def dropColNulls(self, fractionalMetric):
		# fractionalMetric: % nulls in a col before we drop that col
		toDrop = [ key for key, val in self.percentNulls.items() \
			if val > fractionalMetric ]
		self.df.drop(toDrop, axis=1, inplace=True)

		# remove dict elements that were removed from df
		for colName in toDrop:
			if colName in self.numNulls: \
				del self.numNulls[colName]
			if colName in self.percentNulls: \
				del self.percentNulls[colName]

	def dropRowNulls(self, metric):
		# metric: number nulls in a col before we start dropping rows
		toDrop = []
		for key, val in self.numNulls.items():
			if val < metric: # if we are going to remove rows
				nulls = self.df[key].isnull()
				toDrop += list(compress(range(len(nulls)), nulls))
		toDrop = list(set(toDrop))
		self.df.drop(self.df.index[toDrop], inplace=True)
		# will need to recalculate after rows are dropped
		self._nullsDict()