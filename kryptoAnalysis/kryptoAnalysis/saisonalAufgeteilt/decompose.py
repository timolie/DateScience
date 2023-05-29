from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import matplotlib.pyplot as plt



price_data = pd.read_csv("btc-eth-with-weekend.csv")
price_data['Datetime'] = pd.to_datetime(price_data['time'], unit='s')
price_data.index = pd.to_datetime(price_data["Datetime"])
price_data = price_data.drop(['time'], axis=1)
price_data = price_data.drop(['Datetime'], axis=1)


 
price_data = price_data['open_Bitcoin']
print(price_data.head())
price_data.plot()

result = seasonal_decompose(price_data, model='additive', period=4*365)
result.plot()
plt.show()
