import data_modelling
import create_train_test_split
import random_forest_regressor
import plot_feature_importance
import missing_values_as_csv
from data_modelling import drop_price_correlated_data, drop_btc_features

chart_data, data = data_modelling.import_chart_data("../daten/btc_data.csv", False, True, False, True)

drop_price_correlated_data(data)
drop_btc_features(data)

data = data.fillna(0)

data.drop(['close'], axis=1, inplace=True)
data.drop(['high'], axis=1, inplace=True)
data.drop(['low'], axis=1, inplace=True)

independent_train, independent_test, dependent_train, dependent_test, independent, dependent = \
        create_train_test_split.create_train_test_split(data, 0.9)

features, feature_importance, oob_score, accuracy = \
    random_forest_regressor.random_forest_regressor(independent_train, independent_test,
                                                    dependent_train, dependent_test,
                                                    independent, dependent, 10000, 10, 10)

print('Feature Importance: ')
print(feature_importance)
print('oob score: ', oob_score)
print('Genauigkeit der Vorhersage bzgl Testdaten: ', accuracy)
print('')
print('--------------------------------------------------------')
print('')


plot_feature_importance.plot_feature_importance(feature_importance, "TEST")
