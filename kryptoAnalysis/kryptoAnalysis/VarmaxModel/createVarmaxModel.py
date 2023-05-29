import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.varmax import VARMAX

from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys


def execute(data_frame: pd.DataFrame, percentage_of_training_data: float):
    to_row = int(len(data_frame) * percentage_of_training_data)
    training_data = [data_frame[0:to_row]
                     ['open'], data_frame[0:to_row]['close']]

    # training_data = []
    # for column in data_frame.columns:
    #     training_data.append(list(data_frame[0:to_row][column]))

    training_data_exo = data_frame[0:to_row],

    testing_data = list(data_frame[to_row:]['open'])
    model_predictions = []
    n_test_obser = len(testing_data)
    print(training_data)
    for i in range(n_test_obser):

        if (i % 30 == 0):
            sys.stdout.write('\r')
            sys.stdout.write(str(round(100 * float(i)/float(n_test_obser), 2)
                                 ) + "% models created")
            sys.stdout.flush()

        model = VARMAX(endog=training_data, order=(1, 2))
        model_fit = model.fit()
        # output = model_fit.forecast()
        # yhat = output[0]
        # model_predictions.append(yhat)
        # actual_test_value = testing_data[i]
        # training_data.append(actual_test_value)

    # print(model_fit.summary())

    plt.figure(figsize=(15, 9))
    plt.grid(True)

    date_range = data_frame[to_row:].index

    print(len(model_predictions), " testing len", len(testing_data))
    plt.plot(date_range, model_predictions, color='blue', marker='o',
             linestyle='dashed', label='BTC predicted Price')
    plt.plot(date_range, testing_data, color='red', label='BTC Actual Price')
    plt.plot(data_frame[0:to_row]['open'], 'green', label='Train data')

    plt.title('Bitcoin Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # plt.plot(data_frame.index, data_frame['open'])
    # plt.show()
