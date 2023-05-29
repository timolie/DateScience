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

def selectUncorrelatedData(price_data):
    corr = price_data.corr()
    # corr.to_csv('correlation.csv')
    plt.rcParams["figure.figsize"] = (24, 17)
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
        X1 = sm.tools.add_constant(price_data_before)
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
    plt.rcParams["figure.figsize"] = (22, 17)
    plt.title('correlation AFTER selection of uncorrelated columns')
    plt.show()
    return price_data_after

