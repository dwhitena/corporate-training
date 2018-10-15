import pandas as pd
import matplotlib.pyplot as plt
import math

def main():

    # import the data
    data = pd.read_csv('../data/AirPassengers.csv', index_col=0)

    # apply the log transform
    data['value_log'] = data['value'].apply(lambda x: math.log(x))

    # make the series stationary
    data.set_index('time', inplace=True)
    data['value_log_diff'] = data['value_log'].diff()

    data[['value_log', 'value_log_diff']].plot()
    plt.show()

if __name__ == "__main__":
    main()
