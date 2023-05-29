from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import create_train_test_set as tt
import plot_feature_importance as plot


def random_forest(x_train, x_test, y_train, y_test, x, y, n_estimators, max_features, bootstrap, max_depth, min_samples_leaf,
                  min_samples_split, random_state):
    # initialize random forest
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, oob_score=bootstrap,
                                   bootstrap=bootstrap, max_features=max_features, max_depth=max_depth,
                                   min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    model.fit(x_train, y_train)

    # calculate prediction accuracy‚
    prediction = model.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, prediction)
    #print('Genauigkeit des Random Forest Classification Classification (deprecated):', accuracy)
    #print('OOB Score:', model.oob_score_)

    # feature importance
    features = list(x.columns)
    feature_importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    #print("Feature Importance:")
    #print(feature_importance)

    if bootstrap:
        oob = model.oob_score_
    else:
        oob = 0

    return features, feature_importance, oob, accuracy