from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import plot_feature_importance as plot


def random_forest_regressor(x_train, x_test, y_train, y_test, x, y, n_estimators, min_samples_split, min_samples_leaf):
    model = RandomForestRegressor(n_estimators= n_estimators, min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    print("y_test", y_test)
    #accuracy = metrics.accuracy_score(y_test, prediction)
    accuracy = 1

    features = list(x.columns)
    feature_importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

    oob = model.oob_score

    return features, feature_importance, oob, accuracy

