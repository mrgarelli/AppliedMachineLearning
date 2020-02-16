# Import necessary packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tools.Analyzer import Analyzer

df = pd.read_excel("../rsrc/default_of_credit_card_clients.xls", header = 1)

df.rename(columns = {"PAY_0":"PAY_1"}, inplace = True) #renaming mis-named column
df['SEX'] = df['SEX']-1 # change vals of 'sex' to 0,1
df.rename(columns = {'SEX':'FEMALE', "default payment next month":"default"}, inplace = True) # rename col names

for col, pre in zip(["EDUCATION", "MARRIAGE"],["EDU","MAR"]): # get dummies and rename cols for ed and marraige
	df = pd.concat([
		df.drop(
			col,
			axis = "columns"
			),
		pd.get_dummies(
			df[col],
			prefix = pre,
			drop_first = True
			)
		],
		axis = 'columns'
		)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Create tts 
X_train, X_test, y_train, y_test = train_test_split(
	df.drop("default", axis = 'columns'), df['default'],
	test_size = .3, random_state = 1738)

# Instantiate tree and forest models
dt = DecisionTreeClassifier()
bag = BaggingClassifier()
rf = RandomForestClassifier()
et = ExtraTreesClassifier()

# dt.fit(X_train, y_train)
# print("Decision Tree: \n", classification_report(y_test, dt.predict(X_test)), "\n")


# print("-----------")

# bag.fit(X_train, y_train)
# print("Bagging: \n", classification_report(y_test, bag.predict(X_test)), "\n")



# print("-----------")

# rf.fit(X_train, y_train)
# print("Random Forest: \n", classification_report(y_test, rf.predict(X_test)), "\n")

# print("------------")

# et.fit(X_train, y_train)
# print("Extra Trees: \n", classification_report(y_test, et.predict(X_test)), "\n")


from sklearn.metrics import recall_score
def run_comparision():
	criterion = ['gini', 'entropy']
	n_estimators = [5, 10, 20, 50, 100]
	scores = dict()
	i = 0
	for c in criterion:
		for e in n_estimators:
			rf = RandomForestClassifier(n_estimators = e, criterion = c, random_state = 1738)
			rf.fit(X_train, y_train)
			scores[i] = {'recall':recall_score(y_test, rf.predict(X_test)), 'trees' :e, "crit":c}
			i+=1
	print(pd.DataFrame(scores).T)

# run_comparision()

n_estimators = [1,2,3,4,5,6,7,8]
scores2 = dict()
i = 0
for e in n_estimators:
	rf = RandomForestClassifier(n_estimators = e, criterion = 'gini', random_state = 1738)
	rf.fit(X_train, y_train)
	scores2[i] = {'recall':recall_score(y_test, rf.predict(X_test)), 'trees' :e}
	i+=1
print(pd.DataFrame(scores2).T)