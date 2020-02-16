# Import necessary packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tools.Analyzer import Analyzer

# Read in Data
df = pd.read_excel("../rsrc/default_of_credit_card_clients.xls", header = 1)

df.rename(columns = {"PAY_0":"PAY_1"}, inplace = True) #renaming mis-named column

# print(df.head())
# print("Data Shape: " , df.shape, "\n")
# print(df.info())

# defaultPaymentNextMonth = np.array(df['default payment next month'])
# anlzr = Analyzer(defaultPaymentNextMonth)

# print(df['EDUCATION'].value_counts())
# print()
# print(df['MARRIAGE'].value_counts())
# print()
# print(df['SEX'].value_counts())
# print()
# print(df['PAY_1'].value_counts())

# Attribute Information:
# default payment (Yes = 1, No = 0), as the response variable.
#		This study reviewed the literature and used the following
# 		23 variables as explanatory variables: 
# X1: Amount of the given credit (NT dollar)
# 		it includes both the individual consumer credit and his/her
# 		family (supplementary) credit. 
# X2: Gender (1 = male; 2 = female). 
# X3: Education
# 		1 = graduate school
# 		2 = university
# 		3 = high school
# 		4 = others
# X4: Marital status
#		1 = married
# 		2 = single
# 		3 = others 
# X5: Age (year). 
# X6 - X11: History of past payment
# 		We tracked the past monthly payment records
# 		(from April to September, 2005) as follows
# 		X6 = the repayment status in September, 2005
# 		X7 = the repayment status in August, 2005 . . .
# 		X11 = the repayment status in April, 2005
# 		The measurement scale for the repayment status is:
# 			-1 = pay duly
# 			 1 = payment delay for one month
# 			 2 = payment delay for two months . . .
# 			 8 = payment delay for eight months
# 			 9 = payment delay for nine months and above. 
# X12-X17: Amount of bill statement (NT dollar)
# 		X12 = amount of bill statement in September, 2005
# 		X13 = amount of bill statement in August, 2005 . . .
# 		X17 = amount of bill statement in April, 2005. 
# X18-X23: Amount of previous payment (NT dollar)
# 		X18 = amount paid in September, 2005
# 		X19 = amount paid in August, 2005 . . .
# 		X23 = amount paid in April, 2005.

# our key to these values
# for i, col in enumerate(df.columns): print('X' + str(i) + '\t', col)

# for i in [-2,-1,0,1,2,8]:
#     print(df[df['PAY_1']==i][['PAY_1','BILL_AMT1','PAY_AMT1']].head(8), "\n")

### Define function for creating histograms
def pay_hist(df, cols, ymax):
	plt.figure(figsize= (10,7)) # define fig size
	
	for index, col in enumerate(cols): # For each column passed to function
		plt.subplot(2,3, index +1) # plot on new subplot
		plt.ylim(ymax = ymax) # standardize ymax
		plt.hist(df[col]) # create hist
		plt.title(col) # title with column names
	plt.tight_layout() # make sure titles don't overlap

# pay_cols = ["PAY_"+str(n) for n in range(1,7)]
# pay_amt_cols = ['PAY_AMT' + str(n) for n in range(1,7)]

# df[pay_amt_cols].boxplot() # must plot this first
# pay_hist(df, pay_cols, 20000)
# pay_hist(df, pay_amt_cols, 20000)


# df_no_0_pay_amt_1 = df[df["PAY_AMT1"]!=0]
# df_no_0_pay_amt_1["PAY_AMT1"].hist()

# log transformation to look at skewed data
# log_pay_amt1 = np.log10(df_no_0_pay_amt_1["PAY_AMT1"])
# plt.hist(log_pay_amt1)
# plt.title("Log10-Transformed values for 'PAY_AMT1' (Excluding 0s)")

# bill_amt_cols = ['BILL_AMT' + str(n) for n in range(1,7)]
# df[bill_amt_cols].boxplot()
# pay_hist(df, bill_amt_cols, 23000)



# plt.show()

df['SEX'] = df['SEX']-1 # change vals of 'sex' to 0,1

df.rename(columns = {'SEX':'FEMALE', "default payment next month":"default"}, inplace = True) # rename col names

for col, pre in zip(["EDUCATION", "MARRIAGE"],["EDU","MAR"]): # get dummies and rename cols for ed and marraige
    df = pd.concat([
        df.drop(col, axis = "columns"), pd.get_dummies(df[col], prefix = pre, drop_first = True)],
    axis = 'columns')
    
def ginny_impurity(categories):
	tot = sum([sum(li) for li in categories])
	def weighted_node_gi(category):
		catTot = sum(category)
		frcts_sq = [(i/catTot)**2 for i in category]
		gi = 1 - sum(frcts_sq)
		return catTot/tot*gi
	return sum([weighted_node_gi(cat) for cat in categories])
	

# ans = ginny_impurity([[175, 330], [220, 120]])
# print(ans)

# ans = ginny_impurity([[110, 60], [285, 390]])
# print(ans)
