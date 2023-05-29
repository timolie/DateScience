from statsmodels.tsa.stattools import adfuller, kpss
import pandas as pd

def adfuller_test(series, signif=0.05):
    x = adfuller(series, autolag='AIC')

    #using dictionary saves different data types (float, int, boolean)
    output = {'Test Statistic': x[0],
              'P-value': x[1],
              'Number of lags': x[2],
              'Number of observations': x[3],
              f'Reject (signif. level {signif})': x[1] < signif }

    for key, val in x[4].items():
         output[f'Critical value {key}'] = val

    return pd.Series(output)
