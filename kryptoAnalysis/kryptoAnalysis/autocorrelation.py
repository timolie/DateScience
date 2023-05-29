from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from data_modelling import import_chart_data
import matplotlib.pyplot as plt

chart_data, data = import_chart_data("daten/btc_data.csv", False, True, False, True)

data = data.fillna(0)
# Calculate ACF and PACF upto 50 lags
# acf_50 = acf(df.value, nlags=50)
# pacf_50 = pacf(df.value, nlags=50)

# Draw Plot
print('Test')
fig, axes = plt.subplots(1,2,figsize=(16,3), dpi= 100)
plot_acf(data.open.tolist(), lags=50, ax=axes[0])
plot_pacf(data.open.tolist(), lags=50, ax=axes[1])