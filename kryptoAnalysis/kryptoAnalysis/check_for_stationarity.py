from adf import adf_test
from data_modelling import import_chart_data, drop_time, remove_non_features, write_csv_data, read_csv_data
import seaborn as sn
import matplotlib.pyplot as plt
from remove_correlation import selectUncorrelatedData


chart_data, data = import_chart_data("daten/btc_data.csv", False, True, False, True)

data = data.fillna(0)

x = selectUncorrelatedData(data)

# correlation matrix
corrMatrix = data.corr()
plt.rcParams["figure.figsize"] = (24, 17)
sn.heatmap(corrMatrix, annot=True)
plt.show()

# corr matrix before
corrMatrixInitial = chart_data.corr()
plt.rcParams["figure.figsize"] = (24, 17)
sn.heatmap(corrMatrixInitial, annot=True)
plt.show()

adf = data.apply(lambda x: adf_test.adfuller_test(x), axis=0)
adf.insert(0, "", ["Test Statistic", "P-value", "Number of lags", "Number of observations",
                   "Reject (signif. level 0.05)", "Critical value 1% ", "Critical value 5%", "Critical value 10% " ])
print(data.apply(lambda x: adf_test.adfuller_test(x), axis=0))
write_csv_data("daten/adf.csv", adf)
