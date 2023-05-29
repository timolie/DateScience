from xmlrpc.client import boolean
import pandas as pd

MAX_NUMBER = float(9999999999999)
MIN_NUMBER = float(-999999999999)


def read_csv_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, header=0, sep=';')


def write_csv_data(path: str, data_frame: pd.DataFrame) -> pd.DataFrame:
    data_frame.to_csv(path, sep=';', index=False)


def import_chart_data(path: str, include_eth: bool, drop_ma: bool, remove_price: bool, drop_timestamp: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = read_csv_data(path)
    chart_data = data.copy()
    data_modified = create_decision_variables(data)

    data_modified = modify_ema(data_modified)
    data_modified = modify_sma(data_modified)
    data_modified = modify_bb(data_modified)

    data_modified = modify_stock(data_modified)

    if(drop_ma):
        drop_sma_ema(data_modified)

    if(remove_price):
        remove_non_features(data_modified)

    if (include_eth):
        drop_data_before_eth(data_modified)
    else:
        drop_eth_features(data_modified)

    if(drop_timestamp):
        data_modified.set_index("time")
        drop_time(data_modified)

    return chart_data, data_modified


def create_decision_variables(chart):
    data = chart
    # 1D
    for i in range(len(data.index)-1):
        if data.iloc[i, 1] > data.iloc[i+1, 4]:
            d = -1
        else:
            if data.iloc[i, 1] < data.iloc[i+1, 4]:
                d = 1
            else:
                d = 0
        data.loc[data.index[i], 'Decision Variable D'] = d

    # 1D enhanced
    for i in range(len(data.index)-1):
        d = 0
        if data.iloc[i, 1] > data.iloc[i+1, 4] * 1.01 or data.iloc[i, 1] < data.iloc[i+1, 4] * 0.99:
            d = 0
        if data.iloc[i, 1] * 1.01 < data.iloc[i+1, 4] or data.iloc[i, 1] * 1.05 > data.iloc[i+1, 4]:
            d = 1
        if data.iloc[i, 1] * 1.05 < data.iloc[i+1, 4] or data.iloc[i, 1] * 1.10 > data.iloc[i+1, 4]:
            d = 2
        if data.iloc[i, 1] * 1.1 < data.iloc[i + 1, 4]:
            d = 3
        if data.iloc[i, 1] * 1.01 > data.iloc[i+1, 4] or data.iloc[i, 1] * 1.05 < data.iloc[i+1, 4]:
            d = -1
        if data.iloc[i, 1] * 1.05 > data.iloc[i+1, 4] or data.iloc[i, 1] * 1.10 < data.iloc[i+1, 4]:
            d = -2
        if data.iloc[i, 1] * 1.1 > data.iloc[i + 1, 4]:
            d = -3

        data.loc[data.index[i], 'Decision Variable D enhanced'] = d

    for i in range(len(data.index)-30):
        d = 0
        if data.iloc[i, 1] > data.iloc[i+30, 4] * 1.01 or data.iloc[i, 1] < data.iloc[i+30, 4] * 0.975:
            d = 0
        if data.iloc[i, 1] * 1.025 < data.iloc[i+30, 4] or data.iloc[i, 1] * 1.10 > data.iloc[i+30, 4]:
            d = 1
        if data.iloc[i, 1] * 1.10 < data.iloc[i+30, 4] or data.iloc[i, 1] * 1.20 > data.iloc[i+30, 4]:
            d = 2
        if data.iloc[i, 1] * 1.20 < data.iloc[i + 30, 4]:
            d = 3
        if data.iloc[i, 1] * 1.025 > data.iloc[i+30, 4] or data.iloc[i, 1] * 1.10 < data.iloc[i+30, 4]:
            d = -1
        if data.iloc[i, 1] * 1.10 > data.iloc[i+30, 4] or data.iloc[i, 1] * 1.20 < data.iloc[i+30, 4]:
            d = -2
        if data.iloc[i, 1] * 1.20 > data.iloc[i + 30, 4]:
            d = -3

        data.loc[data.index[i], 'Decision Variable M enhanced'] = d

    # 1W
    for i in range(len(data.index)-7):
        if data.iloc[i, 1] > data.iloc[i+7, 4]:
            w = -1
        else:
            if data.iloc[i, 1] < data.iloc[i+7, 4]:
                w = 1
            else:
                w = 0
        data.loc[data.index[i], 'Decision Variable W'] = w

    # 1M
    for i in range(len(data.index) - 30):
        if data.iloc[i, 1] > data.iloc[i+30, 4]:
            m = -1
        else:
            if data.iloc[i, 1] < data.iloc[i+30, 4]:
                m = 1
            else:
                m = 0
        data.loc[data.index[i], 'Decision Variable M'] = m
    return data


def drop_sma_ema(data):
    data.drop(['SMA20'], axis=1, inplace=True)
    data.drop(['SMA50'], axis=1, inplace=True)
    data.drop(['SMA100'], axis=1, inplace=True)
    data.drop(['SMA200'], axis=1, inplace=True)
    data.drop(['EMA 20'], axis=1, inplace=True)
    data.drop(['EMA 50'], axis=1, inplace=True)
    data.drop(['EMA 100'], axis=1, inplace=True)
    data.drop(['EMA 200'], axis=1, inplace=True)


def drop_eth_features(data):
    data.drop(list(data.filter(regex='eth')), axis=1, inplace=True)


def drop_data_before_eth(data):
    data.drop(data[data['time'] <= 1438905600].index, inplace=True)


def remove_non_features(data):
    data.drop(['open'], axis=1, inplace=True)
    data.drop(['close'], axis=1, inplace=True)
    data.drop(['high'], axis=1, inplace=True)
    data.drop(['low'], axis=1, inplace=True)


def drop_time(data):
    data.drop(['time'], axis=1, inplace=True)


def calculate_trends_stocks(data):
    return


def drop_price_correlated_data(data):
    data.drop(['Upper'], axis=1, inplace=True)
    data.drop(['Basis'], axis=1, inplace=True)
    data.drop(['Lower'], axis=1, inplace=True)
    data.drop(['Regression Line'], axis=1, inplace=True)
    data.drop(['Upper Bollinger Band'], axis=1, inplace=True)
    data.drop(['Lower Bollinger Band'], axis=1, inplace=True)
    data.drop(['Bollinger Bands Width'], axis=1, inplace=True)
    data.drop(['ADR'], axis=1, inplace=True)
    data.drop(['Bar Index'], axis=1, inplace=True)
    data.drop(['Signal'], axis=1, inplace=True)
    data.drop(['Histogram'], axis=1, inplace=True)


def drop_btc_features(data):
    data.drop(list(data.filter(regex='btc')), axis=1, inplace=True)


def modify_ema(data_frame):
    rows_to_modify = ['EMA_Closest_TOP', 'EMA_Closest_BOTTOM']
    for row in rows_to_modify:
        data_frame.insert(len(data_frame.columns), row, "0")
    for index, row in data_frame.iterrows():
        ema_to_check = ['EMA 20', 'EMA 50', 'EMA 100', 'EMA 200']
        closest_top, closest_bottom, distance_top, distance_bottom = "0", "0", MAX_NUMBER, MIN_NUMBER
        for i in ema_to_check:
            new_distance = float((row['open'] - row[i]))
            if (new_distance > 0):
                if distance_top > new_distance:
                    distance_top = new_distance
                    closest_top = i
            else:
                if distance_bottom < new_distance:
                    distance_bottom = new_distance
                    closest_bottom = i
        data_frame.at[index, 'EMA_Closest_TOP'] = float(
            closest_top.replace('EMA', ''))
        data_frame.at[index, 'EMA_Closest_BOTTOM'] = float(
            closest_bottom.replace('EMA ', ''))
    return data_frame


def modify_sma(data_frame):
    rows_to_modify = ['SMA_Closest_TOP', 'SMA_Closest_BOTTOM']
    for row in rows_to_modify:
        data_frame.insert(len(data_frame.columns), row, "0")

    for index, row in data_frame.iterrows():
        sma_to_check = ['SMA20', 'SMA50', 'SMA100', 'SMA200']
        closest_top, closest_bottom, distance_top, distance_bottom = "0", "0", MAX_NUMBER, MIN_NUMBER
        for i in sma_to_check:
            new_distance = float((row['open'] - row[i]))
            if (new_distance > 0):
                if distance_top > new_distance:
                    distance_top = new_distance
                    closest_top = i
            else:
                if distance_bottom < new_distance:
                    distance_bottom = new_distance
                    closest_bottom = i
        data_frame.at[index, 'SMA_Closest_TOP'] = float(
            closest_top.replace('SMA', ''))
        data_frame.at[index, 'SMA_Closest_BOTTOM'] = float(
            closest_bottom.replace('SMA', ''))
    return data_frame


def modify_bb(data_frame):
    rows_to_modify = ['BB_Closest']
    for row in rows_to_modify:
        data_frame.insert(len(data_frame.columns), row, "0")

    for index, row in data_frame.iterrows():
        bb_to_check = ['Basis', 'Upper', 'Lower']
        # bb_to_check.pop(0)
        closest,  distance = 0, MAX_NUMBER
        for i in bb_to_check:
            new_distance = abs(float((row['open'] - row[i])))
            if distance > new_distance:
                closest = i
                distance = new_distance
        if closest == 'Basis':
            closest = 0
        elif closest == 'Lower':
            closest = -1
        else:
            closest = 1
        data_frame.at[index, 'BB_Closest'] = closest
    return data_frame


def modify_stock(data_frame):

    translator = {
        'NASDAQ_DLY NDX, 1D open': 'NASDAQ_DLY_NDX_1D_OPEN',
        'OANDA XAUUSD, 1D open': 'OANDA_XAUUSD_1D_OPEN',
        'SP SPX, 1D open': 'SP_SPX_1D_OPEN',
        'TVC DXY, 1D open': 'TVC_DXY_1D_OPEN'
    }
    rows_to_read = ['NASDAQ_DLY NDX, 1D open',
                    'OANDA XAUUSD, 1D open', 'SP SPX, 1D open', 'TVC DXY, 1D open']
    rows_to_modify = ['NASDAQ_DLY_NDX_1D_OPEN_ONE_DAY_TREND',
                      'NASDAQ_DLY_NDX_1D_OPEN_ONE_MONTH_TREND',
                      'OANDA_XAUUSD_1D_OPEN_ONE_DAY_TREND',
                      'OANDA_XAUUSD_1D_OPEN_ONE_MONTH_TREND',
                      'SP_SPX_1D_OPEN_ONE_DAY_TREND',
                      'SP_SPX_1D_OPEN_ONE_MONTH_TREND',
                      'TVC_DXY_1D_OPEN_ONE_DAY_TREND',
                      'TVC_DXY_1D_OPEN_ONE_MONTH_TREND'
                      ]

    for row in rows_to_modify:
        data_frame.insert(len(data_frame.columns), row, "0")

    for index, row in data_frame.iterrows():
        yesterdayTrend, monthAgoTrend = 0, 0
        yesterdayValue, monthAgoValue = 0, 0
        for row_to_read in rows_to_read:
            currentValue = data_frame.iloc[index][row_to_read]
            if (index-1 >= 0):
                yesterdayValue = data_frame.iloc[index-1][row_to_read]
                yesterdayTrend = currentValue / yesterdayValue
                data_frame.at[index, translator[row_to_read] +
                              "_ONE_DAY_TREND"] = yesterdayTrend

            if (index-30 >= 0):
                monthAgoValue = data_frame.iloc[index-30][row_to_read]
                monthAgoTrend = currentValue / monthAgoValue
                data_frame.at[index, translator[row_to_read] +
                              "_ONE_MONTH_TREND"] = monthAgoTrend

    for row in rows_to_read:
        data_frame.drop([row], axis=1, inplace=True)

    return data_frame


'''
def compare_smas(data, include_eth):
    # SMA20
    for i in range(50, len(data.index)):
        if(data.loc[i, 'SMA20'] > data.loc[i, 'SMA50']):
            data.loc[data.index[i], 'SMA20 > SMA50'] = 1
        else:
            data.loc[data.index[i], 'SMA20 > SMA50'] = 0

    for i in range(100, len(data.index)):
        if (data.loc[i, 'SMA20'] > data.loc[i, 'SMA100']):
            data.loc[data.index[i], 'SMA20 > SMA100'] = 1
        else:
            data.loc[data.index[i], 'SMA20 > SMA100'] = 0

    for i in range(200, len(data.index)):
        if (data.loc[i, 'SMA20'] > data.loc[i, 'SMA200']):
            data.loc[data.index[i], 'SMA20 > SMA200'] = 1
        else:
            data.loc[data.index[i], 'SMA20 > SMA200'] = 0

    # SMA50
    for i in range(100, len(data.index)):
        if (data.loc[i, 'SMA50'] > data.loc[i, 'SMA100']):
            data.loc[data.index[i], 'SMA50 > SMA100'] = 1
        else:
            data.loc[data.index[i], 'SMA50 > SMA100'] = 0

    for i in range(200, len(data.index)):
        if (data.loc[i, 'SMA50'] > data.loc[i, 'SMA200']):
            data.loc[data.index[i], 'SMA50 > SMA200'] = 1
        else:
            data.loc[data.index[i], 'SMA50 > SMA200'] = 0

    # SMA100
    for i in range(200, len(data.index)):
        if (data.loc[i, 'SMA100'] > data.loc[i, 'SMA200']):
            data.loc[data.index[i], 'SMA100 > SMA200'] = 1
        else:
            data.loc[data.index[i], 'SMA100 > SMA200'] = 0

    if(include_eth):
        # SMA20_eth
        for i in range(50, len(data.index)):
            if (data.loc[i, 'SMA20_eth'] > data.loc[i, 'SMA50_eth']):
                data.loc[data.index[i], 'SMA20_eth > SMA50_eth'] = 1
            else:
                data.loc[data.index[i], 'SMA20_eth > SMA50_eth'] = 0

        for i in range(100, len(data.index)):
            if (data.loc[i, 'SMA20_eth'] > data.loc[i, 'SMA100_eth']):
                data.loc[data.index[i], 'SMA20_eth > SMA100_eth'] = 1
            else:
                data.loc[data.index[i], 'SMA20_eth > SMA100_eth'] = 0

        for i in range(200, len(data.index)):
            if (data.loc[i, 'SMA20_eth'] > data.loc[i, 'SMA200_eth']):
                data.loc[data.index[i], 'SMA20_eth > SMA200_eth'] = 1
            else:
                data.loc[data.index[i], 'SMA20_eth > SMA200_eth'] = 0

        # SMA50_eth
        for i in range(100, len(data.index)):
            if (data.loc[i, 'SMA50_eth'] > data.loc[i, 'SMA100_eth']):
                data.loc[data.index[i], 'SMA50_eth > SMA100_eth'] = 1
            else:
                data.loc[data.index[i], 'SMA50_eth > SMA100_eth'] = 0

        for i in range(200, len(data.index)):
            if (data.loc[i, 'SMA50_eth'] > data.loc[i, 'SMA200_eth']):
                data.loc[data.index[i], 'SMA50_eth > SMA200_eth'] = 1
            else:
                data.loc[data.index[i], 'SMA50_eth > SMA200_eth'] = 0

        # SMA100_eth
        for i in range(200, len(data.index)):
            if (data.loc[i, 'SMA100_eth'] > data.loc[i, 'SMA200_eth']):
                data.loc[data.index[i], 'SMA100_eth > SMA200_eth'] = 1
            else:
                data.loc[data.index[i], 'SMA100_eth > SMA200_eth'] = 0


def compare_emas(data, include_eth):
    # EMA 20
    for i in range(50, len(data.index)):
        if (data.loc[i, 'EMA 20'] > data.loc[i, 'EMA 50']):
            data.loc[data.index[i], 'EMA 20 > EMA 50'] = 1
        else:
            data.loc[data.index[i], 'EMA 20 > EMA 50'] = 0

    for i in range(100, len(data.index)):
        if (data.loc[i, 'EMA 20'] > data.loc[i, 'EMA 100']):
            data.loc[data.index[i], 'EMA 20 > EMA 100'] = 1
        else:
            data.loc[data.index[i], 'EMA 20 > EMA 100'] = 0

    for i in range(200, len(data.index)):
        if (data.loc[i, 'EMA 20'] > data.loc[i, 'EMA 200']):
            data.loc[data.index[i], 'EMA 20 > EMA 200'] = 1
        else:
            data.loc[data.index[i], 'EMA 20 > EMA 200'] = 0

    # EMA 50
    for i in range(100, len(data.index)):
        if (data.loc[i, 'EMA 50'] > data.loc[i, 'EMA 100']):
            data.loc[data.index[i], 'EMA 50 > EMA 100'] = 1
        else:
            data.loc[data.index[i], 'EMA 50 > EMA 100'] = 0

    for i in range(200, len(data.index)):
        if (data.loc[i, 'EMA 50'] > data.loc[i, 'EMA 200']):
            data.loc[data.index[i], 'EMA 50 > EMA 200'] = 1
        else:
            data.loc[data.index[i], 'EMA 50 > EMA 200'] = 0

    # EMA 100
    for i in range(200, len(data.index)):
        if (data.loc[i, 'EMA 100'] > data.loc[i, 'EMA 200']):
            data.loc[data.index[i], 'EMA 100 > EMA 200'] = 1
        else:
            data.loc[data.index[i], 'EMA 100 > EMA 200'] = 0

    if(include_eth):
        # EMA 20_eth
        for i in range(50, len(data.index)):
            if (data.loc[i, 'EMA 20_eth'] > data.loc[i, 'EMA 50_eth']):
                data.loc[data.index[i], 'EMA 20_eth > EMA 50_eth'] = 1
            else:
                data.loc[data.index[i], 'EMA 20_eth > EMA 50_eth'] = 0

        for i in range(100, len(data.index)):
            if (data.loc[i, 'EMA 20_eth'] > data.loc[i, 'EMA 100_eth']):
                data.loc[data.index[i], 'EMA 20_eth > EMA 100_eth'] = 1
            else:
                data.loc[data.index[i], 'EMA 20_eth > EMA 100_eth'] = 0

        for i in range(200, len(data.index)):
            if (data.loc[i, 'EMA 20_eth'] > data.loc[i, 'EMA 200_eth']):
                data.loc[data.index[i], 'EMA 20_eth > EMA 200_eth'] = 1
            else:
                data.loc[data.index[i], 'EMA 20_eth > EMA 200_eth'] = 0

        # EMA 50_eth
        for i in range(100, len(data.index)):
            if (data.loc[i, 'EMA 50_eth'] > data.loc[i, 'EMA 100_eth']):
                data.loc[data.index[i], 'EMA 50_eth > EMA 100_eth'] = 1
            else:
                data.loc[data.index[i], 'EMA 50_eth > EMA 100_eth'] = 0

        for i in range(200, len(data.index)):
            if (data.loc[i, 'EMA 50_eth'] > data.loc[i, 'EMA 200_eth']):
                data.loc[data.index[i], 'EMA 50_eth > EMA 200_eth'] = 1
            else:
                data.loc[data.index[i], 'EMA 50_eth > EMA 200_eth'] = 0

        # EMA 100_eth
        for i in range(200, len(data.index)):
            if (data.loc[i, 'EMA 100_eth'] > data.loc[i, 'EMA 200_eth']):
                data.loc[data.index[i], 'EMA 100_eth > EMA 200_eth'] = 1
            else:
                data.loc[data.index[i], 'EMA 100_eth > EMA 200_eth'] = 0


def sma_vs_price(data, include_eth):
    for i in range(20, len(data.index)):
        if (data.loc[i, 'open'] > data.loc[i, 'SMA20']):
            data.loc[data.index[i], 'open > SMA20'] = 1
        else:
            data.loc[data.index[i], 'open > SMA20'] = 0

    for i in range(50, len(data.index)):
        if (data.loc[i, 'open'] > data.loc[i, 'SMA50']):
            data.loc[data.index[i], 'open > SMA50'] = 1
        else:
            data.loc[data.index[i], 'open > SMA50'] = 0

    for i in range(100, len(data.index)):
        if (data.loc[i, 'open'] > data.loc[i, 'SMA100']):
            data.loc[data.index[i], 'open > SMA100'] = 1
        else:
            data.loc[data.index[i], 'open > SMA100'] = 0

    for i in range(200, len(data.index)):
        if (data.loc[i, 'open'] > data.loc[i, 'SMA200']):
            data.loc[data.index[i], 'open > SMA200'] = 1
        else:
            data.loc[data.index[i], 'open > SMA200'] = 0

    if(include_eth):
        for i in range(20, len(data.index)):
            if (data.loc[i, 'open'] > data.loc[i, 'SMA20_eth']):
                data.loc[data.index[i], 'open > SMA20_eth'] = 1
            else:
                data.loc[data.index[i], 'open > SMA20_eth'] = 0

        for i in range(50, len(data.index)):
            if (data.loc[i, 'open'] > data.loc[i, 'SMA50_eth']):
                data.loc[data.index[i], 'open > SMA50_eth'] = 1
            else:
                data.loc[data.index[i], 'open > SMA50_eth'] = 0

        for i in range(100, len(data.index)):
            if (data.loc[i, 'open'] > data.loc[i, 'SMA100_eth']):
                data.loc[data.index[i], 'open > SMA100_eth'] = 1
            else:
                data.loc[data.index[i], 'open > SMA100_eth'] = 0

        for i in range(200, len(data.index)):
            if (data.loc[i, 'open'] > data.loc[i, 'SMA200_eth']):
                data.loc[data.index[i], 'open > SMA200_eth'] = 1
            else:
                data.loc[data.index[i], 'open > SMA200_eth'] = 0


def ema_vs_price(data, include_eth):
    for i in range(20, len(data.index)):
        if (data.loc[i, 'open'] > data.loc[i, 'EMA 20']):
            data.loc[data.index[i], 'open > EMA 20'] = 1
        else:
            data.loc[data.index[i], 'open > EMA 20'] = 0

    for i in range(50, len(data.index)):
        if (data.loc[i, 'open'] > data.loc[i, 'EMA 50']):
            data.loc[data.index[i], 'open > EMA 50'] = 1
        else:
            data.loc[data.index[i], 'open > EMA 50'] = 0

    for i in range(100, len(data.index)):
        if (data.loc[i, 'open'] > data.loc[i, 'EMA 100']):
            data.loc[data.index[i], 'open > EMA 100'] = 1
        else:
            data.loc[data.index[i], 'open > EMA 100'] = 0

    for i in range(200, len(data.index)):
        if (data.loc[i, 'open'] > data.loc[i, 'EMA 200']):
            data.loc[data.index[i], 'open > EMA 200'] = 1
        else:
            data.loc[data.index[i], 'open > EMA 200'] = 0

    if (include_eth):
        for i in range(20, len(data.index)):
            if (data.loc[i, 'open'] > data.loc[i, 'EMA 20_eth']):
                data.loc[data.index[i], 'open > EMA 20_eth'] = 1
            else:
                data.loc[data.index[i], 'open > EMA 20_eth'] = 0

        for i in range(50, len(data.index)):
            if (data.loc[i, 'open'] > data.loc[i, 'EMA 50_eth']):
                data.loc[data.index[i], 'open > EMA 50_eth'] = 1
            else:
                data.loc[data.index[i], 'open > EMA 50_eth'] = 0

        for i in range(100, len(data.index)):
            if (data.loc[i, 'open'] > data.loc[i, 'EMA 100_eth']):
                data.loc[data.index[i], 'open > EMA 100_eth'] = 1
            else:
                data.loc[data.index[i], 'open > EMA 100_eth'] = 0

        for i in range(200, len(data.index)):
            if (data.loc[i, 'open'] > data.loc[i, 'EMA 200_eth']):
                data.loc[data.index[i], 'open > EMA 200_eth'] = 1
            else:
                data.loc[data.index[i], 'open > EMA 200_eth'] = 0'''
