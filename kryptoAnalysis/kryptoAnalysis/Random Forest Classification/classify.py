import data_modelling as di
import missing_values_as_csv as mv
from data_modelling import drop_btc_features, drop_price_correlated_data
import execute_random_forest

chart_data, data_modified = di.import_chart_data("../daten/btc_data.csv", False, True, True, True)

missing_values = data_modified.isna().mean().round(4) * 100
mv.write_missing_values_csv(missing_values, 'missing_values.csv')

drop_price_correlated_data(data_modified)

'''
data_modified.drop(['month'], axis=1, inplace=True)
data_modified.drop(['timeToNextHalving'], axis=1, inplace=True)
'''

data_modified = data_modified.fillna(0)


#execute_random_forest.choose_decision_variable(data_modified, "D", False, 0.8)
execute_random_forest.choose_decision_variable(data_modified, "D enhanced", True, 0.8, 1000000)
#execute_random_forest.choose_decision_variable(data_modified, "W", False, 0.8)
#execute_random_forest.choose_decision_variable(data_modified, "M", True, 0.8, 100000)
execute_random_forest.choose_decision_variable(data_modified, "M enhanced", True, 0.8, 1000000)