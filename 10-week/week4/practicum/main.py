import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math

def read_pre_process_series(file):
    """
    Pre-process the series from a CSV file.

    Args:
        file - string containing the file name
    Returns:
        The parsed dataframe containing the series
    """

    # define the date parser
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')

    # read in the data and parse the dates
    data = pd.read_csv(file, parse_dates=['Month'], date_parser=dateparse)

    # rename the columns for prophet
    data.rename(columns={'Month': 'ds', 'Passengers': 'y'}, inplace=True)

    return data

def main():

    # import the data and pre-process the dates
    data = read_pre_process_series('data/AirPassengers.csv')

    # split our data into train and test sets
    split_index = 2*len(data)/3
    train = data.loc[:split_index]
    test = data.loc[split_index+1:]

    # instantiate a new Prophet object
    m = Prophet()

    # fit the prophet model
    m.fit(train)

    # create a forecast
    future = m.make_future_dataframe(periods=len(test), freq = 'M')
    forecast = m.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

    # plot the forecast
    m.plot(forecast)
    plt.show()

    # evaluate a forecast on the test data
    future_for_eval = test[['ds']]
    forecast_for_eval = m.predict(future_for_eval)
    mse = mean_squared_error(forecast_for_eval['yhat'].values, test['y'].values)
    print('\nRMSE: %f' % math.sqrt(mse))

if __name__ == "__main__":
    main()
