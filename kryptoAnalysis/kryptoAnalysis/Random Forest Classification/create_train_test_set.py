from copy import copy
from sklearn.model_selection import train_test_split

#dependant_variable_bl -> Erfolg_BL as dependant variable
def create_train_and_test_set(data, dependant_variable, test_size, shuffle):
    data_new = copy(data)

    if(dependant_variable == "D"):
        data_new.drop(['Decision Variable W'], axis=1, inplace=True)
        data_new.drop(['Decision Variable M'], axis=1, inplace=True)
        data_new.drop(['Decision Variable D enhanced'], axis=1, inplace=True)
        data_new.drop(['Decision Variable M enhanced'], axis=1, inplace=True)
        dependent = data_new['Decision Variable D'].values
        # independent variables
        independent = data_new.drop(labels=['Decision Variable D'], axis=1)

    if (dependant_variable == "D enhanced"):
        data_new.drop(['Decision Variable D'], axis=1, inplace=True)
        data_new.drop(['Decision Variable W'], axis=1, inplace=True)
        data_new.drop(['Decision Variable M'], axis=1, inplace=True)
        data_new.drop(['Decision Variable M enhanced'], axis=1, inplace=True)
        dependent = data_new['Decision Variable D enhanced'].values
        # independent variables
        independent = data_new.drop(labels=['Decision Variable D enhanced'], axis=1)

    if(dependant_variable == "W"):
        data_new.drop(['Decision Variable D'], axis=1, inplace=True)
        data_new.drop(['Decision Variable M'], axis=1, inplace=True)
        data_new.drop(['Decision Variable D enhanced'], axis=1, inplace=True)
        data_new.drop(['Decision Variable M enhanced'], axis=1, inplace=True)
        dependent = data_new['Decision Variable W'].values
        # independent variables
        independent = data_new.drop(labels=['Decision Variable W'], axis=1)

    if(dependant_variable == "M"):
        data_new.drop(['Decision Variable W'], axis=1, inplace=True)
        data_new.drop(['Decision Variable D'], axis=1, inplace=True)
        data_new.drop(['Decision Variable D enhanced'], axis=1, inplace=True)
        data_new.drop(['Decision Variable M enhanced'], axis=1, inplace=True)
        dependent = data_new['Decision Variable M'].values
        # independent variables
        independent = data_new.drop(labels=['Decision Variable M'], axis=1)

    if (dependant_variable == "M enhanced"):
        data_new.drop(['Decision Variable D'], axis=1, inplace=True)
        data_new.drop(['Decision Variable D enhanced'], axis=1, inplace=True)
        data_new.drop(['Decision Variable W'], axis=1, inplace=True)
        data_new.drop(['Decision Variable M'], axis=1, inplace=True)
        dependent = data_new['Decision Variable M enhanced'].values
        # independent variables
        independent = data_new.drop(labels=['Decision Variable M enhanced'], axis=1)


    # split into train- and testset
    if(shuffle):
        independent_train, independent_test, dependent_train, dependent_test = train_test_split(independent,dependent,
                                                                                            test_size=test_size,
                                                                                            random_state=42)
    else:
        independent_train, independent_test, dependent_train, dependent_test = train_test_split(independent, dependent,
                                                                                                test_size=test_size,
                                                                                                shuffle=False)


    return independent_train, independent_test, dependent_train, dependent_test, independent, dependent

