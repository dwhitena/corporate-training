import math

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def pre_process_series(data):
    """
    Difference and apply a log transform to the input
    series data set.

    Args:
        data - time series in dataframe format
    Returns:
        The transformed dataframe
    """
    
    # apply the log transform
    data['value_log'] = data['value'].apply(lambda x: math.log(x))

    # make the series stationary
    data.set_index('time', inplace=True)
    data['value_log_diff'] = data['value_log'].diff()

    return data

def main():

    # import the data
    data = pd.read_csv('../data/AirPassengers.csv', index_col=0)

    # pre-process the data
    data = pre_process_series(data)

    # plot the ACF and PACF
    plot_acf(data[data.value_log_diff.notnull()]['value_log_diff'], lags=20)
    plt.show()
    plot_pacf(data[data.value_log_diff.notnull()]['value_log_diff'], lags=20)
    plt.show()

if __name__ == "__main__":
    main()
