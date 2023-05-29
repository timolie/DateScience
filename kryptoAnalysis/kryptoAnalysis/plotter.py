from pandas.plotting import lag_plot
import pandas as pd
import matplotlib.pyplot as plt
from data_modelling import read_csv_data
import numpy as np


def plot_lag(data_frame):

    x = data_frame['open'].to_numpy()
    y = data_frame['time'].to_numpy()

    print(x, y)

    # s = pd.Series(x)
    # s.plot()
    # data_frame.plot(x='time', y='open')
    data_frame.drop(data_frame.columns.difference(
        ['open', 'time']), 1, inplace=True)
    pd.plotting.lag_plot(data_frame, lag=1)
    plt.show()


plot_lag(read_csv_data('daten/btc_data.csv'))
