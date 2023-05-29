from data_modelling import import_chart_data
from statsmodels.tsa.stattools import grangercausalitytests

chart_data, data = import_chart_data("daten/btc_data.csv", False, True, False, True)

data.drop(['close'], axis=1, inplace=True)
data.drop(['high'], axis=1, inplace=True)
data.drop(['low'], axis=1, inplace=True)
data.drop(list(data.filter(regex='Decision')), axis=1, inplace=True)

data = data.fillna(0)

for c in data.columns:
    print(c)
    grangercausalitytests(data[['open', c]], maxlag=[30])
    print("---------------")
    print("")