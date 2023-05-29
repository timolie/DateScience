from sklearn.model_selection import train_test_split

def create_train_test_split(data, test_size):
    dependent = data['open'].values
    independent = data.drop(labels=['open'], axis=1)
    independent_train, independent_test, dependent_train, dependent_test = train_test_split(independent, dependent,
                                                                                            test_size=test_size,
                                                                                            shuffle=False)

    return independent_train, independent_test, dependent_train, dependent_test, independent, dependent