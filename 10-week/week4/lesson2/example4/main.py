import math

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
import numpy as np

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
    data['value_log_diff'] = data['value_log'].diff()

    return data

def main():

    # import the data
    data = pd.read_csv('../data/AirPassengers.csv', index_col=0)

    # pre-process the data
    data = pre_process_series(data)

    # split our data into train and test sets
    # note - we need to keep the sequence in order
    data_not_null = data[data.value_log_diff.notnull()]
    split_index = 2*len(data)/3
    train = data_not_null.loc[:split_index]
    test = data_not_null.loc[split_index+1:]

    # train autoregression
    model = AR(train['value_log_diff'].values)
    model_fit = model.fit(2)
    print('Lag: %s' % model_fit.k_ar)
    print('Coefficients: %s' % model_fit.params)

    # make a forecast into the "future"
    predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=True)
    predictions_df = pd.DataFrame(predictions, columns=['forecast_log_diff'], index=test.index)
    
    # evaluate the forecast
    test = test.join(predictions_df)
    test.loc[:, 'forecast'] = test.loc[:, 'forecast_log_diff'].cumsum() + data.loc[split_index-1, 'value_log']
    test.loc[:, 'forecast'] = test.loc[:, 'forecast'].apply(lambda x: np.exp(x))
    mse = mean_squared_error(test['value'].values, test['forecast'].values)
    print('RMSE: %f' % math.sqrt(mse))

    # retrieve the fit model values from the model_fit object
    model_fit_df = pd.DataFrame([np.nan, np.nan] + model_fit.fittedvalues.tolist(), 
            columns=['model_fit_log_diff'], index=train.index)
    train = train.join(model_fit_df)
    train.loc[:, 'model_fit'] = train.loc[:, 'model_fit_log_diff'].cumsum() + data.loc[2, 'value_log']
    train.loc[:, 'model_fit'] = train.loc[:, 'model_fit'].apply(lambda x: np.exp(x))

    # visualize the forecast, model fit, and original data
    data = data.join(test['forecast'])
    data = data.join(train['model_fit'])
    data.set_index('time')[['value', 'model_fit', 'forecast']].plot()
    plt.show()

if __name__ == "__main__":
    main()
