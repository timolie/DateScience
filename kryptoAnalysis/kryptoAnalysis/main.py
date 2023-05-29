from hashlib import new
from pandas import DataFrame
import data_modelling as dm


dataframe = dm.read_csv_data("daten/btc_data.csv")

dataframe = dm.modify_stock(dataframe)

print(dataframe["SP_SPX_1D_OPEN_ONE_MONTH_TREND"])

#dm.write_csv_data('daten/test_neu.csv', dataframe)
