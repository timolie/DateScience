import create_train_test_set as tt
import random_forest_classification as rf
import plot_feature_importance as plot

def choose_decision_variable(data, decision_variable, shuffle, split, n_estimators=1000):
    independent_train, independent_test, dependent_train, dependent_test, independent, dependent = \
        tt.create_train_and_test_set(data, decision_variable, split, shuffle)


    features, feature_importance, oob_score, accuracy = rf.random_forest(independent_train, independent_test,
                                                                         dependent_train, dependent_test, independent,
                                                                         dependent, n_estimators, 'log2', True, 15, 5, 20, 42)

    print('Decision Variable: ', decision_variable)
    print('Feature Importance: ')
    print(feature_importance)
    print('oob score: ', oob_score)
    print('Genauigkeit der Vorhersage bzgl Testdaten: ', accuracy)
    print('')
    print('--------------------------------------------------------')
    print('')

    plot.plot_feature_importance(feature_importance, 'Decision Variable: ' + decision_variable + ' Shuffle: ' +
                                 str(shuffle))