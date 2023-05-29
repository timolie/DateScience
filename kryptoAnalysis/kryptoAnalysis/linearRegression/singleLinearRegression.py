import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from scipy import stats
from scipy.stats import kurtosis, skew

# read data
path = "btc-eth-with-weekend.csv"
price_data = pd.read_csv(path)
print("first data head", price_data.head())

# index data
price_data.index = pd.to_datetime(price_data["time"])
price_data = price_data.drop(['time'], axis=1)
print("index data head", price_data.head())

print("price data describe \n ", "-" * 200, price_data['timeToNextHalving'].describe())



price_data['absoluteDifferenceToHalving'] = abs(abs(price_data['timeToNextHalving']-710)-710)
print("absolute difference \n", price_data['absoluteDifferenceToHalving'].to_string())
price_data["timeToNextHalving"] = price_data['absoluteDifferenceToHalving']
# check the data types
#print("check data types", price_data.dtypes)

# rename columns
new_column_names = {'year': 'Year'}
price_data = price_data.rename(columns = new_column_names)
#print("new column names", price_data.head())

# verify no nan values
# delete na values if there are any
#print("check for nans", price_data.isna().any())
price_data = price_data.dropna()


#print("print column names", price_data.columns)
# build a scatter plot
x = price_data['timeToNextHalving']
y = price_data['open_Bitcoin']
plt.plot(x,y, 'o', color='cadetblue', label="Daily Price")
plt.title('Price vs Days since or to next Halving')
plt.xlabel('timeToNextHalving')
plt.ylabel('open Price')
plt.legend()
plt.show()

# measure correlation
#correlations = (price_data.corr())
#print("correlations", correlations)

# create statistical summary
#print("check statistics of data with data describe", price_data.describe())

# check for outliers and skewness
# tells only if there are outliers or skewness but does not tell how much because this depends on data size
#hist = price_data.hist(column=["open_Bitcoin", "timeToNextHalving", "SMA20_Bitcoin", "TVC DXY, 1D low"], grid=False, color="cadetblue")
#plt.show()

timeToNextHalving_kurtosis = kurtosis(price_data['timeToNextHalving'], fisher= True)
timeToNextHalving_skew = skew(price_data['timeToNextHalving'])

#
#print("timeToNextHalving Kurtosis: {:.2}".format(timeToNextHalving_kurtosis))
#print("timeToNextHalving Skew: {:.2}".format(timeToNextHalving_skew))

#print("TimeToNextHalving")
#print(stats.kurtosistest(price_data["timeToNextHalving"]))
#print(stats.skewtest(price_data["timeToNextHalving"]))

# build linear regression model in one variable
Y = price_data[['open_Bitcoin']] # output variable
X = price_data[['timeToNextHalving']] # input variable
# split X and Y into X_
# todo: make split for time series data
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=.20, random_state=1)
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)
intercept = regression_model.intercept_[0]
coefficient = regression_model.coef_[0][0]
# model y = intercept + coefficient * data
#print("the coefficient for our model is {:.2}".format(coefficient))
#print("the intercept for our model is {:.4}".format(intercept))
prediction = regression_model.predict([[130]])
predicted_value = prediction[0][0]
#print("The predicted value is {:4}".format(predicted_value))

# predict multiple values
y_predict = regression_model.predict(X_test)
#print(y_predict[:5])


# make more predictions
# define input
X2 = sm.add_constant(X)
# create OLS model
model = sm.OLS(Y,X2)
# fit the data
est = model.fit()

# make some confidence intervals 95% by default
print("conf intervall \n", est.conf_int())
# hypothesis testing
# H0 the is not relationsship between feature and price --> coefficient = 0
# H1 there is a relation --> coefficient != 0
# estimate the p value
# value < 0.05 --> reject H0
# think there is a relation between halving time and price
print("estimate p values \n", est.pvalues)

# check how good model is performing
# MAE : mean of the absolute value of the error
# MSE: punishes big errors more
# RMSE: square root of MSE

# mean squared error
model_mse = mean_squared_error(y_test, y_predict)

# mean absolute error
model_mae = mean_absolute_error(y_test, y_predict)

# root mean squared error
model_rmse = math.sqrt(model_mae)
print("MSE {:.3}".format(model_mse))
print("MAE {:.3}".format(model_mae))
print("RMSE {:.3}".format(model_rmse))

# r squared
# square of the correlation
# the higher the better the data is fit
model_r2 = r2_score(y_test, y_predict)
# predicts very very bad
print("R2 {:.2}".format(model_r2))

# summary of model
# very very bad results
print(est.summary())

# plot residuals
(y_test - y_predict).hist(grid=False, color="royalblue")
plt.title('Model Residuals')
plt.show()

# plot outpus
plt.scatter(X_test, y_test, color="gainsboro", label="Price")
plt.plot(X_test, y_predict, color="royalblue" , linewidth=3, linestyle= "-", label="Regression Line")
plt.title("Linear Regression BTC Price vs TimeToHalving")
plt.ylabel('BTC Price')
plt.xlabel("TimeToHalving")
plt.legend()
plt.show()

# save model for later use
import pickle
# pickle the model
with open("my_linear_regression.sav", 'wb') as f:
    pickle.dump(regression_model,f)
# load it back in
with open("my_linear_regression.sav", "rb") as f:
    regression_model_2 = pickle.load(f)

print(regression_model_2.predict([[150]]))