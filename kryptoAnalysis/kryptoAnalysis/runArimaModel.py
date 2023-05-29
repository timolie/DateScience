import ArimaModel.createArimaModel as ArimaModel
import VarmaxModel.createVarmaxModel as VarmaxModel
from data_modelling import import_chart_data, read_csv_data


chart_data, data_modified = import_chart_data(
    "daten/btc_data.csv", False, True, False, True)

print(data_modified)

# data_frame = read_csv_data("daten/btc_data.csv")

# print(data_frame.index)
# ArimaModel.execute(data_modified, 0.8)
VarmaxModel.execute(data_modified, 0.8)
