import pandas as pd
import matplotlib.pyplot as plt

def main():

    # import the data
    data = pd.read_csv('../data/AirPassengers.csv', index_col=0)

    # make the series stationary
    data.set_index('time', inplace=True)
    data['value_diff'] = data['value'].diff()

    data.plot()
    plt.show()

if __name__ == "__main__":
    main()
