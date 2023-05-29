# see https://github.com/areed1192/sigma_coding_youtube/blob/master/python/python-data-science/machine-learning/multi-linear-regression/Machine%20Learning%20-%20Multi%20Linear%20Regression%20Analysis.ipynb
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.stats import diagnostic as diag
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# read data
path = "btc-eth-with-weekend.csv"
price_data = pd.read_csv(path)

price_data['Datetime'] = pd.to_datetime(price_data['time'], unit='s')
price_data.index = pd.to_datetime(price_data["Datetime"])
price_data = price_data.drop(['time'], axis=1)
price_data = price_data.drop(['Datetime'], axis=1)

#print("index data head", price_data.head())
price_data = price_data.drop(['date'], axis=1)
new_column_names = {'year': 'Year',
                    '%EBitcoin_Bitcoin': 'btc ex/on chain trans vol',
                    'SOPR' : 'Spent Output Profit Ratio',
                    'hashColor_Bitcoin' : 'Btc sopr momentum onchain'}
price_data = price_data.rename(columns=new_column_names)

print(price_data['Spent Output Profit Ratio'].head())


#price_data = price_data[0:100][:]
#print(price_data.describe())
#price_data = price_data.head(100)
"""
change absolute data to relative data

absoluteColumns = ['month', 'dayInMonth', 'wekkday', 'year', 'timeToNextHalving']
for column in price_data.columns:
    if column not in absoluteColumns:
        price_data[column] = price_data[column].pct_change()
price_data = price_data.iloc[1: , :]
#print(price_data.head())
"""

# need the data without multicollinearity
# test for multicollinearity


"""
However, we should be more systematic in our approach to removing highly correlated variables. 
One method we can use is the variance_inflation_factor which is a measure of how much a particular
variable is contributing to the standard error in the regression model. When significant 
multicollinearity exists, the variance inflation factor will be huge for the variables
in the calculation.
"""


def selectUncorrelatedData(price_data):
    corr = price_data.corr()
    # corr.to_csv('correlation.csv')
    sns.heatmap(corr, cmap="RdBu")
    plt.title('correlation before selection of uncorrelated columns')
    plt.show()
    # delete highly correlated data
    price_data_before = price_data
    X1 = sm.tools.add_constant(price_data_before)
    series_before = pd.Series([variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])], index= X1.columns)


    #pd.plotting.scatter_matrix(price_data_before.iloc[:, 0:8], alpha=1, figsize =(30,20))
    #plt.title('bad features correlation shown')
    #plt.show()

    print('Data before')
    print("-"*100)
    print(series_before.head(50))
    #print((series_before < 5).to_string())
    price_data_after = price_data.loc[:, series_before < 5]
    i = 1
    while not price_data_after.empty:
        price_data_before = price_data_after
        X1 = sm.tools.add_constant(price_data_before) # added 1 zu spalte was verfahren numerisch stabiler macht
        series_before = pd.Series([variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])], index=X1.columns)
        price_data_after = price_data_before.loc[:, series_before < 5].iloc[:, 1:]
        X2 = sm.tools.add_constant(price_data_after)
        series_after = pd.Series([variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])], index=X2.columns)
        #print('-'*200)
        #print(series_after.to_string())
        i += 1
        if i > 2:
           break
    #print((price_data_after.columns))
    print('Data after')
    print("-"*100)
    print(series_after.head(50))



    corr = price_data_after.corr()
    #corr.to_csv('correlation.csv')
    sns.heatmap(corr, cmap="RdBu", label="heatmap after deleting columns")
    plt.title('correlation AFTER selection of uncorrelated columns')
    plt.show()
    return price_data_after



# defne the plot
#pd.plotting.scatter_matrix(price_data_after, alpha=1, figsize =(30,20))
#plt.title('good features shown')
#plt.show()


# inspect data and detect outliers
# on original data set
description = price_data.describe()
#print(description)

description.loc['+3_std'] = description.loc['mean'] + (description.loc['std']*3)
description.loc['-3_std'] = description.loc['mean'] - (description.loc['std']*3)
#print(description.to_string())
# filter outliers
# zscore = number of standard deviations data is away from mean
price_data_remove = price_data[(np.abs(stats.zscore(price_data)) < 3).all(axis=1)]
# print removed rows
#print(price_data.index.difference(price_data_remove.index))

price_data_after = selectUncorrelatedData(price_data)
# now we can work with price_data_remove or price_data whether we want extrem values to be deleted
# define input and output variable
X = price_data_after
# if Logistic Regression
Y = (np.sign(price_data['open_Bitcoin'].pct_change()))
# if Linear Regression
Y = price_data['open_Bitcoin']
Y = Y.fillna(0)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=1)

# logistic regression
#regression_model = LogisticRegression()

# linear regression
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)
# Get multiple predictions
y_predict = regression_model.predict(X_test)
y_predict = y_predict
y_test = y_test
print("#"*200)
print("prediction \n", *y_predict)
print("test \n", *y_test)
plt.fill_between([i*10 for i in range(len(y_predict))], y_predict, label="prediction", color="orange")
plt.plot([i*10 for i in range(len(y_predict))], y_test, label="real price", color= 'blue', linewidth=0.3)
plt.title('Multiple Linear Regression vs Price')
plt.legend()
plt.show()







"""
# build model with better libary for evaluation
# define our intput
X2 = sm.add_constant(X)
# create a OLS model
model = sm.OLS(Y, X2)
# fit the data
est = model.fit()
# Run the Breusch-Pagan test
_, pval, __, f_pval = diag.het_breuschpagan(est.resid, est.model.exog)
print(pval, f_pval)
print('-'*100)

# todo: problem because there is heterosecdasticity
# print the results of the test
if pval > 0.05:
    print("For the Breusch-Pagan's Test")
    print("The p-value was {:.4}".format(pval))
    print("We fail to reject the null hypthoesis, so there is no heterosecdasticity.")

else:
    print("For the Breusch-Pagan's Test")
    print("The p-value was {:.4}".format(pval))
    print("We reject the null hypthoesis, so there is heterosecdasticity.")

# todo: problem because test of autocorrelation fails


import pylab
# check for the normality of the residuals
sm.qqplot(est.resid, line='s')
pylab.show()

# also check that the mean of the residuals is approx. 0.
mean_residuals = sum(est.resid)/ len(est.resid)
print("The mean of the residuals is {:.4}".format(mean_residuals))
"""


"""
import math
# calculate the mean squared error
model_mse = mean_squared_error(y_test, y_predict)

# calculate the mean absolute error
model_mae = mean_absolute_error(y_test, y_predict)

# calulcate the root mean squared error
model_rmse =  math.sqrt(model_mse)

# display the output
print("MSE {:.3}".format(model_mse))
print("MAE {:.3}".format(model_mae))
print("RMSE {:.3}".format(model_rmse))

#r suqared test
model_r2 = r2_score(y_test, y_predict)
print("R2: {:.2}".format(model_r2))
"""
"""
# make some confidence intervals, 95% by default
print(est.conf_int())

# estimate the p-values
print(est.pvalues)

# print out a summary
#print(est.summary())

X = price_data_after.drop('%EBitcoin_Bitcoin', axis=1)
Y = price_data['open_Bitcoin']
# Split X and y into X_
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=1)

# create a Linear Regression model object
regression_model = LinearRegression()

# pass through the X_train & y_train data set
regression_model.fit(X_train, y_train)



X2 = sm.add_constant(X)
# create a OLS model
model = sm.OLS(Y, X2)
# fit the data
est = model.fit()
print(est.summary())




import pickle

# pickle the model
with open('my_mulitlinear_regression.sav','wb') as f:
     pickle.dump(regression_model, f)

# load it back in
with open('my_mulitlinear_regression.sav', 'rb') as pickle_file:
     regression_model_2 = pickle.load(pickle_file)

# new prediction
regression_model_2.predict([X_test.loc[2002]])


"""



